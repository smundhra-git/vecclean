#include "cleaner.hpp"
#include <algorithm>
#include <regex>
#include <sstream>
#include <unordered_set>
#include <cstring>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <thread>
#include <queue>
#include <condition_variable>
#include <exception>
#include <stdexcept>

// Include specific C++ standard library features for SIMD
#if defined(VECCLEAN_HAS_SSE2) || defined(VECCLEAN_HAS_AVX2)
#include <x86intrin.h>
#endif

namespace vecclean {

// Global memory allocator for high-frequency operations
static thread_local std::unique_ptr<MemoryPool> g_thread_memory_pool;

// ============================================================================
// MemoryPool Implementation
// ============================================================================

MemoryPool::MemoryPool(size_t pool_size) : pool_size_(pool_size), current_offset_(0), allocations_(0) {
    // Allocate aligned memory for SIMD operations
    pool_ = std::aligned_alloc(64, pool_size_);
    if (!pool_) {
        throw std::bad_alloc();
    }
    std::memset(pool_, 0, pool_size_);
}

MemoryPool::~MemoryPool() {
    if (pool_) {
        std::free(pool_);
    }
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Align size to requested alignment
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    
    size_t current = current_offset_.load();
    if (current + aligned_size > pool_size_) {
        throw std::bad_alloc();
    }
    
    void* ptr = static_cast<char*>(pool_) + current;
    current_offset_.store(current + aligned_size);
    allocations_.fetch_add(1);
    
    return ptr;
}

void MemoryPool::deallocate(void* ptr) {
    // Simple pool - no individual deallocation, only reset
    (void)ptr; // Suppress unused parameter warning
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_offset_.store(0);
    allocations_.store(0);
}

size_t MemoryPool::get_used_memory() const {
    return current_offset_.load();
}

size_t MemoryPool::get_total_memory() const {
    return pool_size_;
}

size_t MemoryPool::get_allocation_count() const {
    return allocations_.load();
}

// ============================================================================
// PerformanceProfiler Implementation
// ============================================================================

PerformanceProfiler& PerformanceProfiler::instance() {
    static PerformanceProfiler profiler;
    return profiler;
}

void PerformanceProfiler::start_profiling(const std::string& function_name, const std::unordered_map<std::string, std::string>& metadata) {
    if (!enabled_.load()) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    ProfileEntry& entry = active_profiles_[function_name];
    entry.function_name = function_name;
    entry.start_time = std::chrono::high_resolution_clock::now();
    entry.metadata = metadata;
    
    // Get memory usage if available
    if (g_thread_memory_pool) {
        entry.memory_before = g_thread_memory_pool->get_used_memory();
    }
}

void PerformanceProfiler::end_profiling(const std::string& function_name) {
    if (!enabled_.load()) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_profiles_.find(function_name);
    if (it != active_profiles_.end()) {
        it->second.end_time = std::chrono::high_resolution_clock::now();
        
        // Get memory usage if available
        if (g_thread_memory_pool) {
            it->second.memory_after = g_thread_memory_pool->get_used_memory();
        }
        
        profiles_.push_back(std::move(it->second));
        active_profiles_.erase(it);
    }
}

std::vector<PerformanceProfiler::ProfileEntry> PerformanceProfiler::get_profile_data() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiles_;
}

void PerformanceProfiler::clear_profile_data() {
    std::lock_guard<std::mutex> lock(mutex_);
    profiles_.clear();
    active_profiles_.clear();
}

void PerformanceProfiler::set_enabled(bool enabled) {
    enabled_.store(enabled);
}

bool PerformanceProfiler::is_enabled() const {
    return enabled_.load();
}

// ============================================================================
// ScopedProfiler Implementation
// ============================================================================

ScopedProfiler::ScopedProfiler(const std::string& function_name, const std::unordered_map<std::string, std::string>& metadata)
    : function_name_(function_name) {
    PerformanceProfiler::instance().start_profiling(function_name_, metadata);
}

ScopedProfiler::~ScopedProfiler() {
    PerformanceProfiler::instance().end_profiling(function_name_);
}

// ============================================================================
// LanguageDetector Implementation
// ============================================================================

LanguageDetector::Language LanguageDetector::detect_language(const std::string& text) const {
    // Simple language detection based on common words and character patterns
    std::unordered_map<Language, int> scores;
    
    // Convert to lowercase for analysis
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    // English indicators
    if (lower_text.find(" the ") != std::string::npos || 
        lower_text.find(" and ") != std::string::npos ||
        lower_text.find(" of ") != std::string::npos) {
        scores[Language::ENGLISH] += 3;
    }
    
    // Spanish indicators
    if (lower_text.find(" el ") != std::string::npos || 
        lower_text.find(" la ") != std::string::npos ||
        lower_text.find(" de ") != std::string::npos) {
        scores[Language::SPANISH] += 3;
    }
    
    // French indicators
    if (lower_text.find(" le ") != std::string::npos || 
        lower_text.find(" la ") != std::string::npos ||
        lower_text.find(" du ") != std::string::npos) {
        scores[Language::FRENCH] += 3;
    }
    
    // German indicators
    if (lower_text.find(" der ") != std::string::npos || 
        lower_text.find(" die ") != std::string::npos ||
        lower_text.find(" das ") != std::string::npos) {
        scores[Language::GERMAN] += 3;
    }
    
    // Find language with highest score
    Language best_language = Language::UNKNOWN;
    int best_score = 0;
    for (const auto& pair : scores) {
        if (pair.second > best_score) {
            best_score = pair.second;
            best_language = pair.first;
        }
    }
    
    return best_language;
}

std::unordered_set<std::string> LanguageDetector::get_stopwords(Language lang) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = stopwords_cache_.find(lang);
    if (it != stopwords_cache_.end()) {
        return it->second;
    }
    
    std::unordered_set<std::string> stopwords;
    
    switch (lang) {
        case Language::ENGLISH:
            stopwords = {"a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
                        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to",
                        "was", "will", "with", "the", "this", "but", "they", "have", "had", "what",
                        "said", "each", "which", "she", "do", "how", "their", "if", "up", "out",
                        "many", "then", "them", "these", "so", "some", "her", "would", "make",
                        "like", "into", "him", "time", "two", "more", "go", "no", "way", "could",
                        "my", "than", "first", "water", "been", "call", "who", "oil", "its", "now",
                        "find", "long", "down", "day", "did", "get", "come", "made", "may", "part"};
            break;
        case Language::SPANISH:
            stopwords = {"el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo",
                        "le", "da", "su", "por", "son", "con", "para", "al", "una", "ser", "son",
                        "me", "ya", "muy", "del", "los", "si", "mi", "puede", "todo", "esta",
                        "le", "ha", "por", "o", "puede", "tener", "de", "tu", "mucho", "en",
                        "ah", "mismo", "algo"};
            break;
        case Language::FRENCH:
            stopwords = {"le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour",
                        "dans", "ce", "son", "une", "sur", "avec", "ne", "se", "pas", "tout", "plus",
                        "par", "grand", "en", "le", "bien", "être", "à", "il", "avoir", "savoir",
                        "aller", "pouvoir", "falloir", "voir", "en", "faire", "dire", "tout",
                        "son", "autre"};
            break;
        case Language::GERMAN:
            stopwords = {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des",
                        "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als", "auch",
                        "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach", "wird",
                        "bei", "noch", "wie", "einem", "über", "einen", "so", "zum", "war", "haben",
                        "nur", "oder", "aber", "vor", "zur", "bis", "mehr", "durch", "man", "sein",
                        "wurde", "sei", "haben"};
            break;
        default:
            break;
    }
    
    // Cache the result
    const_cast<LanguageDetector*>(this)->stopwords_cache_[lang] = stopwords;
    return stopwords;
}

std::string LanguageDetector::get_language_specific_rules(Language lang) const {
    switch (lang) {
        case Language::ENGLISH:
            return "preserve_contractions,smart_quotes_to_ascii";
        case Language::SPANISH:
            return "preserve_accents,normalize_inverted_punctuation";
        case Language::FRENCH:
            return "preserve_accents,normalize_cedilla";
        case Language::GERMAN:
            return "preserve_umlauts,normalize_eszett";
        default:
            return "default_rules";
    }
}

// ============================================================================
// StreamingProcessor Implementation
// ============================================================================

StreamingProcessor::StreamingProcessor(const CleaningConfig& config, size_t chunk_size)
    : config_(config), chunk_size_(chunk_size), processor_(std::make_unique<TextProcessor>(config)) {
}

void StreamingProcessor::process_stream(
    std::function<std::string()> input_reader,
    std::function<void(const std::string&)> output_writer) {
    
    ScopedProfiler profiler("StreamingProcessor::process_stream");
    
    std::string buffer;
    std::string chunk;
    
    try {
        while (!(chunk = input_reader()).empty()) {
            buffer += chunk;
            
            // Process complete chunks
            while (buffer.size() >= chunk_size_) {
                std::string to_process = buffer.substr(0, chunk_size_);
                buffer = buffer.substr(chunk_size_);
                
                std::string cleaned = processor_->clean_text(to_process);
                output_writer(cleaned);
                
                // Update stats
                const auto& chunk_stats = processor_->get_last_stats();
                stats_.input_length += chunk_stats.input_length;
                stats_.output_length += chunk_stats.output_length;
                stats_.processing_time_ms += chunk_stats.processing_time_ms;
            }
        }
        
        // Process remaining buffer
        if (!buffer.empty()) {
            std::string cleaned = processor_->clean_text(buffer);
            output_writer(cleaned);
            
            const auto& final_stats = processor_->get_last_stats();
            stats_.input_length += final_stats.input_length;
            stats_.output_length += final_stats.output_length;
            stats_.processing_time_ms += final_stats.processing_time_ms;
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Streaming processing failed: " + std::string(e.what()));
    }
}

void StreamingProcessor::process_file(const std::string& input_file, const std::string& output_file) {
    std::ifstream input(input_file, std::ios::binary);
    std::ofstream output(output_file, std::ios::binary);
    
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open input file: " + input_file);
    }
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }
    
    auto input_reader = [&input, this]() -> std::string {
        std::string chunk(chunk_size_, '\0');
        input.read(&chunk[0], chunk_size_);
        chunk.resize(input.gcount());
        return chunk;
    };
    
    auto output_writer = [&output](const std::string& data) {
        output.write(data.c_str(), data.size());
    };
    
    process_stream(input_reader, output_writer);
}

// ============================================================================
// Fuzzy String Matching Implementation
// ============================================================================

namespace fuzzy {

FuzzyMatch compute_fuzzy_similarity(const std::string& text1, const std::string& text2) {
    FuzzyMatch result;
    result.edit_distance = edit_distance(text1, text2);
    result.similarity = std::max(0.0, 1.0 - static_cast<double>(result.edit_distance) / std::max(text1.length(), text2.length()));
    return result;
}

double jaccard_similarity(const std::string& text1, const std::string& text2) {
    auto ngrams1 = generate_ngrams(text1, 3);
    auto ngrams2 = generate_ngrams(text2, 3);
    
    std::unordered_set<std::string> set1(ngrams1.begin(), ngrams1.end());
    std::unordered_set<std::string> set2(ngrams2.begin(), ngrams2.end());
    
    size_t intersection = 0;
    for (const auto& ngram : set1) {
        if (set2.count(ngram)) {
            intersection++;
        }
    }
    
    size_t union_size = set1.size() + set2.size() - intersection;
    return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
}

double cosine_similarity(const std::string& text1, const std::string& text2) {
    auto ngrams1 = generate_ngrams(text1, 3);
    auto ngrams2 = generate_ngrams(text2, 3);
    
    std::unordered_map<std::string, int> freq1, freq2;
    for (const auto& ngram : ngrams1) freq1[ngram]++;
    for (const auto& ngram : ngrams2) freq2[ngram]++;
    
    double dot_product = 0.0;
    double norm1 = 0.0, norm2 = 0.0;
    
    std::unordered_set<std::string> all_ngrams;
    for (const auto& p : freq1) all_ngrams.insert(p.first);
    for (const auto& p : freq2) all_ngrams.insert(p.first);
    
    for (const auto& ngram : all_ngrams) {
        int f1 = freq1[ngram];
        int f2 = freq2[ngram];
        dot_product += f1 * f2;
        norm1 += f1 * f1;
        norm2 += f2 * f2;
    }
    
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
}

size_t edit_distance(const std::string& text1, const std::string& text2) {
    size_t m = text1.length();
    size_t n = text2.length();
    
    std::vector<std::vector<size_t>> dp(m + 1, std::vector<size_t>(n + 1));
    
    for (size_t i = 0; i <= m; i++) dp[i][0] = i;
    for (size_t j = 0; j <= n; j++) dp[0][j] = j;
    
    for (size_t i = 1; i <= m; i++) {
        for (size_t j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
            }
        }
    }
    
    return dp[m][n];
}

std::vector<std::string> generate_ngrams(const std::string& text, size_t n) {
    std::vector<std::string> ngrams;
    if (text.length() < n) return ngrams;
    
    for (size_t i = 0; i <= text.length() - n; i++) {
        ngrams.push_back(text.substr(i, n));
    }
    return ngrams;
}

} // namespace fuzzy

// ============================================================================
// SIMD Utilities Implementation
// ============================================================================

namespace util {
namespace simd {

bool has_sse2_support() {
#ifdef VECCLEAN_HAS_SSE2
    return true;
#else
    return false;
#endif
}

bool has_avx2_support() {
#ifdef VECCLEAN_HAS_AVX2
    return true;
#else
    return false;
#endif
}

#ifdef VECCLEAN_HAS_SSE2
std::string normalize_whitespace_sse2(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    
    const char* data = text.c_str();
    size_t len = text.length();
    size_t i = 0;
    bool prev_was_space = true;
    
    // Process 16 characters at a time using SSE2
    for (; i + 16 <= len; i += 16) {
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
        
        // Check for whitespace characters (space, tab, newline, carriage return)
        __m128i spaces = _mm_cmpeq_epi8(chunk, _mm_set1_epi8(' '));
        __m128i tabs = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('\t'));
        __m128i newlines = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('\n'));
        __m128i carriages = _mm_cmpeq_epi8(chunk, _mm_set1_epi8('\r'));
        
        __m128i whitespace = _mm_or_si128(_mm_or_si128(spaces, tabs), _mm_or_si128(newlines, carriages));
        
        // Convert to bitmask for efficient checking
        int mask = _mm_movemask_epi8(whitespace);
        
        // Process each character in the chunk
        for (int j = 0; j < 16; j++) {
            char c = data[i + j];
            bool is_space = (mask & (1 << j)) != 0;
            
            if (is_space) {
                if (!prev_was_space) {
                    result += ' ';
                    prev_was_space = true;
                }
            } else {
                result += c;
                prev_was_space = false;
            }
        }
    }
    
    // Process remaining characters
    for (; i < len; i++) {
        char c = data[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_was_space) {
                result += ' ';
                prev_was_space = true;
            }
        } else {
            result += c;
            prev_was_space = false;
        }
    }
    
    // Remove trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}
#else
std::string normalize_whitespace_sse2(const std::string& text) {
    // Fallback to non-SIMD implementation
    return normalize_whitespace_avx2(text);
}
#endif

#ifdef VECCLEAN_HAS_AVX2
std::string normalize_whitespace_avx2(const std::string& text) {
    if (text.empty()) return text;
    
    std::string result;
    result.reserve(text.length());
    
    const char* data = text.data();
    size_t len = text.length();
    bool prev_was_space = true;
    
    // Process 32 characters at a time using AVX2
    size_t i = 0;
    for (; i + 31 < len; i += 32) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        
        // Create masks for different whitespace characters
        __m256i spaces = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(' '));
        __m256i tabs = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\t'));
        __m256i newlines = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\n'));
        __m256i carriages = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8('\r'));
        
        __m256i whitespace = _mm256_or_si256(_mm256_or_si256(spaces, tabs), _mm256_or_si256(newlines, carriages));
        
        // Store to buffer and process each character
        alignas(32) uint8_t buf[32];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), whitespace);
        
        for (int j = 0; j < 32; j++) {
            char c = data[i + j];
            bool is_space = (buf[j] != 0);
            
            if (is_space) {
                if (!prev_was_space) {
                    result += ' ';
                    prev_was_space = true;
                }
            } else {
                result += c;
                prev_was_space = false;
            }
        }
    }
    
    // Process remaining characters
    for (; i < len; i++) {
        char c = data[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_was_space) {
                result += ' ';
                prev_was_space = true;
            }
        } else {
            result += c;
            prev_was_space = false;
        }
    }
    
    // Remove trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}
#else
std::string normalize_whitespace_avx2(const std::string& text) {
    // Fallback to scalar implementation
    std::string result;
    result.reserve(text.length());
    
    bool prev_was_space = true;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_was_space) {
                result += ' ';
                prev_was_space = true;
            }
        } else {
            result += c;
            prev_was_space = false;
        }
    }
    
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}
#endif

std::string remove_control_chars_sse2(const std::string& text) {
    // Implementation similar to whitespace normalization
    std::string result;
    result.reserve(text.length());
    
    for (unsigned char c : text) {
        if ((c >= 32 && c <= 126) || c == '\t' || c == '\n' || c == '\r' || c > 127) {
            result += c;
        }
    }
    
    return result;
}

std::string remove_control_chars_avx2(const std::string& text) {
    // Implementation similar to whitespace normalization
    return remove_control_chars_sse2(text); // Use same implementation for now
}

} // namespace simd

// ============================================================================
// Work Stealing Thread Pool Implementation
// ============================================================================

struct WorkStealingThreadPool::Implementation {
    std::vector<std::thread> workers;
    std::vector<std::queue<std::function<void()>>> queues;
    std::vector<std::unique_ptr<std::mutex>> queue_mutexes;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    std::atomic<size_t> active_tasks{0};
    
    explicit Implementation(size_t num_threads) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        
        queues.resize(num_threads);
        queue_mutexes.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            queue_mutexes.emplace_back(std::make_unique<std::mutex>());
        }
        
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i, num_threads] {
                while (!stop.load()) {
                    std::function<void()> task;
                    
                    // Try to get task from own queue first
                    {
                        std::unique_lock<std::mutex> lock(*queue_mutexes[i]);
                        if (!queues[i].empty()) {
                            task = std::move(queues[i].front());
                            queues[i].pop();
                            active_tasks.fetch_add(1);
                        }
                    }
                    
                    // If no task in own queue, try to steal from others
                    if (!task) {
                        for (size_t j = 1; j < num_threads; ++j) {
                            size_t target = (i + j) % num_threads;
                            std::unique_lock<std::mutex> lock(*queue_mutexes[target]);
                            if (!queues[target].empty()) {
                                task = std::move(queues[target].front());
                                queues[target].pop();
                                active_tasks.fetch_add(1);
                                break;
                            }
                        }
                    }
                    
                    if (task) {
                        task();
                        active_tasks.fetch_sub(1);
                        condition.notify_all();
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                }
            });
        }
    }
    
    ~Implementation() {
        stop.store(true);
        condition.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    template<typename F>
    auto submit(F&& f) -> std::future<std_compat::result_of_t<F>> {
        using return_type = std_compat::result_of_t<F>;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> result = task->get_future();
        
        // Round-robin task assignment
        static thread_local size_t next_queue = 0;
        size_t queue_id = next_queue++ % queues.size();
        
        {
            std::lock_guard<std::mutex> lock(*queue_mutexes[queue_id]);
            queues[queue_id].emplace([task]() { (*task)(); });
        }
        
        condition.notify_one();
        return result;
    }
    
    void wait_for_all() {
        std::unique_lock<std::mutex> lock(*queue_mutexes[0]);
        condition.wait(lock, [this] {
            return active_tasks.load() == 0 && 
                   std::all_of(queues.begin(), queues.end(), [](const auto& q) { return q.empty(); });
        });
    }
    
    size_t get_thread_count() const {
        return workers.size();
    }
};

WorkStealingThreadPool::WorkStealingThreadPool(size_t num_threads) 
    : impl_(std::make_unique<Implementation>(num_threads)) {
}

WorkStealingThreadPool::~WorkStealingThreadPool() = default;

template<typename F>
auto WorkStealingThreadPool::submit(F&& f) -> std::future<std_compat::result_of_t<F>> {
    return impl_->submit(std::forward<F>(f));
}

void WorkStealingThreadPool::wait_for_all() {
    impl_->wait_for_all();
}

size_t WorkStealingThreadPool::get_thread_count() const {
    return impl_->get_thread_count();
}

} // namespace util

// ============================================================================
// TextProcessor Implementation
// ============================================================================

// Private implementation class
struct TextProcessor::Implementation {
    std::unordered_set<std::string> stopwords_cache;
    std::string working_buffer;
    std::unique_ptr<util::WorkStealingThreadPool> thread_pool;
    
    Implementation() {
        thread_pool = std::make_unique<util::WorkStealingThreadPool>();
    }
};

TextProcessor::TextProcessor(const CleaningConfig& config) 
    : config_(config), impl_(std::make_unique<Implementation>()) {
    
    // Initialize memory pool if enabled
    if (config_.use_memory_pool) {
        memory_pool_ = std::make_unique<MemoryPool>(config_.memory_pool_size);
        if (!g_thread_memory_pool) {
            g_thread_memory_pool = std::make_unique<MemoryPool>(config_.memory_pool_size);
        }
    }
    
    // Initialize language detector if enabled
    if (config_.enable_language_detection) {
        language_detector_ = std::make_unique<LanguageDetector>();
    }
}

std::string TextProcessor::clean_text(const std::string& input_text) {
    ScopedProfiler profiler("TextProcessor::clean_text");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::string result = input_text;
    size_t original_length = input_text.length();
    
    try {
        // Step 1: Unicode normalization
        if (config_.normalize_unicode) {
            result = normalize_unicode_internal(result);
        }
        
        // Step 2: Remove control characters
        if (config_.remove_control_chars) {
            if (config_.use_simd) {
                result = remove_control_chars_simd(result);
            } else {
                result = remove_control_chars_internal(result);
            }
        }
        
        // Step 3: Standardize punctuation
        if (config_.standardize_punctuation) {
            result = standardize_punctuation_internal(result);
        }
        
        // Step 4: Normalize whitespace
        if (config_.normalize_whitespace) {
            if (config_.use_simd) {
                result = normalize_whitespace_simd(result);
            } else {
                result = normalize_whitespace(result);
            }
        }
        
        // Step 5: Remove empty lines and trim
        if (config_.remove_empty_lines) {
            std::istringstream iss(result);
            std::string line;
            std::vector<std::string> lines;
            while (std::getline(iss, line)) {
                // Trim line
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (!line.empty() && static_cast<int>(line.length()) >= config_.min_line_length) {
                    lines.push_back(line);
                }
            }
            result = "";
            for (size_t i = 0; i < lines.size(); ++i) {
                if (i > 0) result += "\n";
                result += lines[i];
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Text cleaning failed: " + std::string(e.what()));
    }
    
    // Update stats
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    last_stats_.input_length = original_length;
    last_stats_.output_length = result.length();
    last_stats_.chars_normalized = original_length - result.length();
    last_stats_.processing_time_ms = duration.count() / 1000.0;
    last_stats_.compression_ratio = original_length > 0 ? 
        static_cast<double>(result.length()) / original_length : 1.0;
    
    if (memory_pool_) {
        last_stats_.peak_memory_usage = memory_pool_->get_used_memory();
        last_stats_.memory_allocations = memory_pool_->get_allocation_count();
    }
    
    return result;
}

std::string TextProcessor::normalize_whitespace(const std::string& text) {
    ScopedProfiler profiler("TextProcessor::normalize_whitespace");
    
    if (config_.use_simd) {
        return normalize_whitespace_simd(text);
    }
    
    std::string result;
    result.reserve(text.length());
    
    bool prev_was_space = true; // Start as true to trim leading spaces
    
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_was_space) {
                result += ' ';
                prev_was_space = true;
            }
        } else {
            result += c;
            prev_was_space = false;
        }
    }
    
    // Remove trailing space if any
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}

std::string TextProcessor::strip_stopwords(
    const std::string& text, 
    const std::unordered_set<std::string>& stopwords,
    bool preserve_semantic) {
    
    ScopedProfiler profiler("TextProcessor::strip_stopwords");
    
    std::istringstream iss(text);
    std::string word;
    std::string result;
    
    try {
        while (iss >> word) {
            // Remove punctuation from word for stopword checking
            std::string cleaned_word = word;
            cleaned_word.erase(std::remove_if(cleaned_word.begin(), cleaned_word.end(), 
                [](char c) { return std::ispunct(static_cast<unsigned char>(c)); }), 
                cleaned_word.end());
            
            // Convert to lowercase for case-insensitive comparison
            std::string lower_word = cleaned_word;
            std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
            
            // Check if word should be preserved
            bool should_preserve = false;
            if (preserve_semantic && is_semantic_token(cleaned_word)) {
                should_preserve = true;
            }
            
            // Keep word if it's not a stopword or should be preserved
            if (should_preserve || stopwords.find(lower_word) == stopwords.end()) {
                if (!result.empty()) {
                    result += " ";
                }
                result += word;
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Stopword removal failed: " + std::string(e.what()));
    }
    
    return result;
}

std::vector<uint64_t> TextProcessor::hash_sentences(const std::vector<std::string>& sentences) {
    ScopedProfiler profiler("TextProcessor::hash_sentences");
    
    std::vector<uint64_t> hashes;
    hashes.reserve(sentences.size());
    
    try {
        if (config_.parallel_processing && sentences.size() > 100) {
            // Parallel processing for large datasets
            std::vector<std::future<uint64_t>> futures;
            
            for (const auto& sentence : sentences) {
                futures.push_back(impl_->thread_pool->submit([this, sentence]() {
                    std::string cleaned = sentence;
                    
                    // Convert to lowercase for case-insensitive hashing
                    std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
                    
                    // Remove punctuation
                    cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), 
                        [](char c) { return std::ispunct(static_cast<unsigned char>(c)); }), 
                        cleaned.end());
                    
                    // Normalize whitespace
                    cleaned = normalize_whitespace(cleaned);
                    
                    return compute_fast_hash(cleaned);
                }));
            }
            
            for (auto& future : futures) {
                hashes.push_back(future.get());
            }
            
        } else {
            // Sequential processing
            for (const auto& sentence : sentences) {
                std::string cleaned = sentence;
                
                // Convert to lowercase for case-insensitive hashing
                std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
                
                // Remove punctuation
                cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), 
                    [](char c) { return std::ispunct(static_cast<unsigned char>(c)); }), 
                    cleaned.end());
                
                // Normalize whitespace
                cleaned = normalize_whitespace(cleaned);
                
                // Hash the cleaned sentence
                hashes.push_back(compute_fast_hash(cleaned));
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Sentence hashing failed: " + std::string(e.what()));
    }
    
    return hashes;
}

std::vector<size_t> TextProcessor::dedup_chunks(
    const std::vector<std::string>& chunks,
    double exact_threshold,
    double similarity_threshold) {
    
    ScopedProfiler profiler("TextProcessor::dedup_chunks");
    
    if (config_.parallel_processing && chunks.size() > 1000) {
        return dedup_chunks_parallel(chunks, exact_threshold, similarity_threshold);
    }
    
    std::vector<size_t> keep_indices;
    std::unordered_set<uint64_t> seen_hashes;
    std::vector<std::string> kept_chunks;
    
    try {
        for (size_t i = 0; i < chunks.size(); ++i) {
            const std::string& chunk = chunks[i];
            
            // Compute hash for exact deduplication
            uint64_t hash = compute_fast_hash(chunk);
            
            // Check exact hash match
            if (seen_hashes.find(hash) != seen_hashes.end()) {
                continue; // Skip exact duplicate
            }
            
            // Check fuzzy similarity with existing chunks
            bool is_duplicate = false;
            for (const auto& kept_chunk : kept_chunks) {
                double similarity = compute_similarity(chunk, kept_chunk);
                if (similarity >= similarity_threshold) {
                    is_duplicate = true;
                    break;
                }
            }
            
            if (!is_duplicate) {
                keep_indices.push_back(i);
                seen_hashes.insert(hash);
                kept_chunks.push_back(chunk);
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Chunk deduplication failed: " + std::string(e.what()));
    }
    
    return keep_indices;
}

std::vector<size_t> TextProcessor::dedup_chunks_parallel(
    const std::vector<std::string>& chunks,
    double exact_threshold,
    double similarity_threshold) {
    
    ScopedProfiler profiler("TextProcessor::dedup_chunks_parallel");
    
    // Note: exact_threshold is reserved for future exact hash matching optimization
    (void)exact_threshold;  // Suppress unused parameter warning
    
    std::vector<size_t> keep_indices;
    std::unordered_set<uint64_t> seen_hashes;
    std::mutex indices_mutex, hashes_mutex;
    
    // Compute hashes in parallel
    std::vector<uint64_t> hashes(chunks.size());
    std::vector<std::future<void>> hash_futures;
    
    const size_t chunk_size = std::max(size_t(1), chunks.size() / impl_->thread_pool->get_thread_count());
    
    for (size_t start = 0; start < chunks.size(); start += chunk_size) {
        size_t end = std::min(start + chunk_size, chunks.size());
        
        hash_futures.push_back(impl_->thread_pool->submit([this, &chunks, &hashes, start, end]() {
            for (size_t i = start; i < end; ++i) {
                hashes[i] = compute_fast_hash(chunks[i]);
            }
        }));
    }
    
    // Wait for hash computation to complete
    for (auto& future : hash_futures) {
        future.get();
    }
    
    // Sequential deduplication (parallelizing this is complex due to dependencies)
    std::vector<std::string> kept_chunks;
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (seen_hashes.find(hashes[i]) != seen_hashes.end()) {
            continue;
        }
        
        bool is_duplicate = false;
        for (const auto& kept_chunk : kept_chunks) {
            double similarity = compute_similarity(chunks[i], kept_chunk);
            if (similarity >= similarity_threshold) {
                is_duplicate = true;
                break;
            }
        }
        
        if (!is_duplicate) {
            keep_indices.push_back(i);
            seen_hashes.insert(hashes[i]);
            kept_chunks.push_back(chunks[i]);
        }
    }
    
    return keep_indices;
}

void TextProcessor::update_config(const CleaningConfig& new_config) {
    config_ = new_config;
    
    // Reinitialize components if needed
    if (new_config.use_memory_pool && !memory_pool_) {
        memory_pool_ = std::make_unique<MemoryPool>(new_config.memory_pool_size);
    }
    
    if (new_config.enable_language_detection && !language_detector_) {
        language_detector_ = std::make_unique<LanguageDetector>();
    }
}

// SIMD-optimized operations
std::string TextProcessor::normalize_whitespace_simd(const std::string& text) {
#ifdef VECCLEAN_HAS_AVX2
    return util::simd::normalize_whitespace_avx2(text);
#elif defined(VECCLEAN_HAS_SSE2)
    return util::simd::normalize_whitespace_sse2(text);
#else
    return normalize_whitespace(text);
#endif
}

std::string TextProcessor::remove_control_chars_simd(const std::string& text) {
#ifdef VECCLEAN_HAS_AVX2
    return util::simd::remove_control_chars_avx2(text);
#elif defined(VECCLEAN_HAS_SSE2)
    return util::simd::remove_control_chars_sse2(text);
#else
    return remove_control_chars_internal(text);
#endif
}

// Internal helper methods (existing implementations with error handling)
std::string TextProcessor::normalize_unicode_internal(const std::string& text) {
    try {
        std::string result = text;
        
        // Apply custom replacements if configured
        for (const auto& replacement : config_.custom_replacements) {
            std::regex pattern(replacement.first);
            result = std::regex_replace(result, pattern, replacement.second);
        }
        
        // Standard Unicode normalization
        std::regex smart_single_quotes(u8"['']");
        result = std::regex_replace(result, smart_single_quotes, "'");
        
        std::regex smart_double_quotes(u8"[""]");
        result = std::regex_replace(result, smart_double_quotes, "\"");
        
        std::regex dashes(u8"[–—]");
        result = std::regex_replace(result, dashes, "-");
        
        std::regex ellipsis(u8"[…]");
        result = std::regex_replace(result, ellipsis, "...");
        
        return result;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Unicode normalization failed: " + std::string(e.what()));
    }
}

std::string TextProcessor::standardize_punctuation_internal(const std::string& text) {
    try {
        std::string result = text;
        
        // Standardize multiple periods to single periods
        std::regex multiple_periods("\\.{2,}");
        result = std::regex_replace(result, multiple_periods, ".");
        
        // Standardize multiple question marks
        std::regex multiple_questions("\\?{2,}");
        result = std::regex_replace(result, multiple_questions, "?");
        
        // Standardize multiple exclamation marks
        std::regex multiple_exclamations("!{2,}");
        result = std::regex_replace(result, multiple_exclamations, "!");
        
        // Add space after punctuation if missing
        std::regex punct_no_space("([.!?])([A-Za-z])");
        result = std::regex_replace(result, punct_no_space, "$1 $2");
        
        return result;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Punctuation standardization failed: " + std::string(e.what()));
    }
}

std::string TextProcessor::remove_control_chars_internal(const std::string& text) {
    std::string result;
    result.reserve(text.length());
    
    for (unsigned char c : text) {
        // Keep printable ASCII chars, tabs, newlines, and carriage returns
        if ((c >= 32 && c <= 126) || c == '\t' || c == '\n' || c == '\r' || c > 127) {
            result += c;
        }
    }
    
    return result;
}

std::vector<std::string> TextProcessor::split_sentences_internal(const std::string& text) {
    std::vector<std::string> sentences;
    
    try {
        // Simple sentence splitting on common terminators
        std::regex sentence_regex(R"([.!?]+\s+)");
        std::sregex_token_iterator iter(text.begin(), text.end(), sentence_regex, -1);
        std::sregex_token_iterator end;
        
        for (; iter != end; ++iter) {
            std::string sentence = iter->str();
            // Trim whitespace
            sentence.erase(0, sentence.find_first_not_of(" \t\r\n"));
            sentence.erase(sentence.find_last_not_of(" \t\r\n") + 1);
            
            if (!sentence.empty() && static_cast<int>(sentence.length()) >= config_.min_text_length) {
                sentences.push_back(sentence);
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Sentence splitting failed: " + std::string(e.what()));
    }
    
    return sentences;
}

bool TextProcessor::is_semantic_token(const std::string& word) {
    // Consider words semantic if they're longer than 2 chars and not just punctuation
    if (word.length() <= 2) return false;
    
    // Check if word contains at least one alphanumeric character
    bool has_alnum = false;
    for (char c : word) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            has_alnum = true;
            break;
        }
    }
    
    return has_alnum;
}

uint64_t TextProcessor::compute_fast_hash(const std::string& text) {
    // Enhanced FNV-1a hash implementation with better distribution
    const uint64_t fnv_prime = 1099511628211ULL;
    const uint64_t fnv_offset_basis = 14695981039346656037ULL;
    
    uint64_t hash = fnv_offset_basis;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= fnv_prime;
    }
    
    // Additional mixing for better distribution
    hash ^= hash >> 32;
    hash *= 0x9e3779b97f4a7c15ULL;
    hash ^= hash >> 32;
    
    return hash;
}

double TextProcessor::compute_similarity(const std::string& text1, const std::string& text2) {
    if (text1 == text2) return 1.0;
    if (text1.empty() || text2.empty()) return 0.0;
    
    // Use enhanced fuzzy matching for better similarity detection
    return fuzzy::jaccard_similarity(text1, text2);
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

namespace util {

uint64_t fast_hash(const std::string& text) {
    const uint64_t fnv_prime = 1099511628211ULL;
    const uint64_t fnv_offset_basis = 14695981039346656037ULL;
    
    uint64_t hash = fnv_offset_basis;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= fnv_prime;
    }
    
    return hash;
}

double jaccard_similarity(const std::string& text1, const std::string& text2) {
    return fuzzy::jaccard_similarity(text1, text2);
}

bool is_whitespace_only(const std::string& text) {
    return std::all_of(text.begin(), text.end(), [](unsigned char c) {
        return std::isspace(c);
    });
}

std::string normalize_unicode(const std::string& text, UnicodeForm form) {
    // Basic Unicode normalization - this would ideally use ICU library
    std::string result = text;
    
    switch (form) {
        case UnicodeForm::NFC:
            // Canonical decomposition followed by canonical composition
            break;
        case UnicodeForm::NFD:
            // Canonical decomposition
            break;
        case UnicodeForm::NFKC:
            // Compatibility decomposition followed by canonical composition
            break;
        case UnicodeForm::NFKD:
            // Compatibility decomposition
            break;
    }
    
    return result;
}

} // namespace util

} // namespace vecclean

// ============================================================================
// C-style API implementations
// ============================================================================

extern "C" {

int vecclean_init() {
    try {
        // Check for SIMD support and log capabilities
        bool simd_available = false;
        #ifdef VECCLEAN_HAS_SSE2
            simd_available = true;
        #endif
        
        // Note: simd_available is for future capability reporting
        (void)simd_available;  // Suppress unused variable warning
        
        // Initialize profiler
        vecclean::PerformanceProfiler::instance().set_enabled(true);
        
        return 0; // Success
    } catch (...) {
        return -1; // Error
    }
}

int vecclean_clean_text_simple(const char* input, char* output, size_t output_size) {
    if (!input || !output || output_size == 0) return -1;
    
    try {
        vecclean::CleaningConfig config;
        vecclean::TextProcessor processor(config);
        std::string result = processor.clean_text(std::string(input));
        
        if (result.length() >= output_size) return -1;
        
        std::strcpy(output, result.c_str());
        return static_cast<int>(result.length());
    } catch (...) {
        return -1;
    }
}

int vecclean_normalize_whitespace(const char* input, char* output, size_t output_size) {
    if (!input || !output || output_size == 0) return -1;
    
    try {
        vecclean::CleaningConfig config;
        vecclean::TextProcessor processor(config);
        std::string result = processor.normalize_whitespace(std::string(input));
        
        if (result.length() >= output_size) return -1;
        
        std::strcpy(output, result.c_str());
        return static_cast<int>(result.length());
    } catch (...) {
        return -1;
    }
}

int vecclean_strip_stopwords(
    const char* input,
    const char* const* stopwords,
    size_t stopword_count,
    char* output,
    size_t output_size) {
    
    if (!input || !stopwords || !output || output_size == 0) return -1;
    
    try {
        std::unordered_set<std::string> stopword_set;
        for (size_t i = 0; i < stopword_count; ++i) {
            stopword_set.insert(std::string(stopwords[i]));
        }
        
        vecclean::CleaningConfig config;
        vecclean::TextProcessor processor(config);
        std::string result = processor.strip_stopwords(std::string(input), stopword_set, true);
        
        if (result.length() >= output_size) return -1;
        
        std::strcpy(output, result.c_str());
        return static_cast<int>(result.length());
    } catch (...) {
        return -1;
    }
}

uint64_t vecclean_hash_sentence(const char* sentence) {
    if (!sentence) return 0;
    
    try {
        vecclean::CleaningConfig config;
        vecclean::TextProcessor processor(config);
        std::vector<std::string> sentences = {std::string(sentence)};
        std::vector<uint64_t> hashes = processor.hash_sentences(sentences);
        return hashes.empty() ? 0 : hashes[0];
    } catch (...) {
        return 0;
    }
}

uint32_t vecclean_get_simd_capabilities() {
    uint32_t capabilities = 0;
    
    #ifdef VECCLEAN_HAS_SSE2
        capabilities |= 0x1;  // SSE2 bit
    #endif
    
    #ifdef VECCLEAN_HAS_AVX2
        capabilities |= 0x2;  // AVX2 bit
    #endif
    
    return capabilities;
}

void vecclean_cleanup() {
    try {
        // Cleanup any global resources
        vecclean::PerformanceProfiler::instance().clear_profile_data();
    } catch (...) {
        // Ignore cleanup errors
    }
}

} // extern "C"
