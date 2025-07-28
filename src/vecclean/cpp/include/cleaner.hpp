/**
 * VecClean C++ Text Processing Library
 * 
 * High-performance C++ implementations for CPU-intensive text processing operations.
 * This library provides significant speed improvements over pure Python implementations
 * for operations like text normalization, deduplication, and stopword removal.
 * 
 * Key features:
 * - Unicode-aware text normalization
 * - Fast whitespace and punctuation standardization
 * - Efficient stopword removal with configurable preservations
 * - High-speed sentence and chunk hashing for deduplication
 * - SIMD-optimized string operations where possible
 * - Thread-safe implementations for parallel processing
 * - Memory pool allocation for high-frequency operations
 * - Streaming API for very large documents
 * - Profiling hooks for performance monitoring
 * - Language-specific text processing rules
 * 
 * Performance targets:
 * - Text normalization: ~6x faster than Python
 * - Deduplication hashing: ~8x faster than Python
 * - Stopword removal: ~5x faster than Python
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <future>

// C++17 compatibility - use invoke_result instead of result_of
#if __cplusplus >= 201703L
#include <type_traits>
namespace std_compat {
    template<typename F, typename... Args>
    using result_of_t = std::invoke_result_t<F, Args...>;
}
#else
namespace std_compat {
    template<typename F, typename... Args>
    using result_of_t = typename std::result_of<F(Args...)>::type;
}
#endif

// SIMD support detection
#ifdef __SSE2__
#include <emmintrin.h>
#define VECCLEAN_HAS_SSE2 1
#endif

#ifdef __AVX2__
#include <immintrin.h>
#define VECCLEAN_HAS_AVX2 1
#endif

namespace vecclean {

/**
 * Configuration structure for text cleaning operations.
 * Maps to Python configuration for consistency.
 */
struct CleaningConfig {
    // Unicode normalization options
    bool normalize_unicode = true;
    std::string unicode_form = "NFC";  // NFC, NFD, NFKC, NFKD
    
    // Whitespace and formatting
    bool normalize_whitespace = true;
    bool remove_extra_newlines = true;
    bool trim_lines = true;
    
    // Punctuation and character standardization
    bool standardize_punctuation = true;
    bool standardize_quotes = true;
    bool standardize_dashes = true;
    bool remove_control_chars = true;
    
    // Content filtering
    bool remove_empty_lines = true;
    int min_line_length = 3;
    int max_line_length = 10000;
    int min_text_length = 10;  // Added for sentence splitting
    
    // Performance options
    bool use_simd = true;
    bool parallel_processing = true;
    int thread_count = 0;  // 0 = auto-detect
    
    // Language-specific processing
    bool enable_language_detection = false;
    
    // Custom Unicode normalization rules
    std::unordered_map<std::string, std::string> custom_replacements;
    
    // Memory management
    bool use_memory_pool = true;
    size_t memory_pool_size = 64 * 1024 * 1024;
};

/**
 * Statistics from text cleaning operations.
 * Provides detailed metrics for performance monitoring.
 */
struct CleaningStats {
    // Input statistics
    size_t input_length = 0;
    size_t input_lines = 0;
    
    // Output statistics  
    size_t output_length = 0;
    size_t output_lines = 0;
    
    // Operations performed
    size_t chars_normalized = 0;
    size_t whitespace_normalized = 0;
    size_t punctuation_standardized = 0;
    size_t lines_removed = 0;
    size_t control_chars_removed = 0;
    
    // Performance metrics
    double processing_time_ms = 0.0;
    double throughput_mb_per_sec = 0.0;
    
    // Compression achieved
    double compression_ratio = 0.0;  // output_length / input_length
    
    // Memory usage
    size_t peak_memory_usage = 0;
    size_t memory_allocations = 0;
    
    void reset() {
        *this = CleaningStats{};
    }
};

// Forward declarations
class TextProcessor;
class MemoryPool;
class StreamingProcessor;
class PerformanceProfiler;
class LanguageDetector;

/**
 * Memory pool for high-frequency operations.
 * Reduces allocation overhead by pre-allocating large blocks.
 */
class MemoryPool {
public:
    explicit MemoryPool(size_t pool_size = 64 * 1024 * 1024);
    ~MemoryPool();
    
    void* allocate(size_t size, size_t alignment = 8);
    void deallocate(void* ptr);
    void reset();
    
    size_t get_used_memory() const;
    size_t get_total_memory() const;
    size_t get_allocation_count() const;
    
private:
    void* pool_;
    size_t pool_size_;
    std::atomic<size_t> current_offset_;
    std::atomic<size_t> allocations_;
    mutable std::mutex mutex_;
};

/**
 * Performance profiler for monitoring operation performance.
 */
class PerformanceProfiler {
public:
    struct ProfileEntry {
        std::string function_name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        size_t memory_before;
        size_t memory_after;
        std::unordered_map<std::string, std::string> metadata;
    };
    
    static PerformanceProfiler& instance();
    
    void start_profiling(const std::string& function_name, const std::unordered_map<std::string, std::string>& metadata = {});
    void end_profiling(const std::string& function_name);
    std::vector<ProfileEntry> get_profile_data() const;
    void clear_profile_data();
    void set_enabled(bool enabled);
    bool is_enabled() const;
    
private:
    PerformanceProfiler() = default;
    mutable std::mutex mutex_;
    std::vector<ProfileEntry> profiles_;
    std::unordered_map<std::string, ProfileEntry> active_profiles_;
    std::atomic<bool> enabled_{false};
};

/**
 * RAII profiler for automatic timing.
 */
class ScopedProfiler {
public:
    explicit ScopedProfiler(const std::string& function_name, const std::unordered_map<std::string, std::string>& metadata = {});
    ~ScopedProfiler();
    
private:
    std::string function_name_;
};

/**
 * Language detection and processing rules.
 */
class LanguageDetector {
public:
    enum class Language {
        UNKNOWN,
        ENGLISH,
        SPANISH,
        FRENCH,
        GERMAN,
        ITALIAN,
        PORTUGUESE,
        RUSSIAN,
        CHINESE,
        JAPANESE,
        ARABIC
    };
    
    Language detect_language(const std::string& text) const;
    std::unordered_set<std::string> get_stopwords(Language lang) const;
    std::string get_language_specific_rules(Language lang) const;
    
private:
    std::unordered_map<Language, std::unordered_set<std::string>> stopwords_cache_;
    mutable std::mutex cache_mutex_;
};

/**
 * Streaming processor for very large documents.
 * Processes text in chunks to handle memory constraints.
 */
class StreamingProcessor {
public:
    explicit StreamingProcessor(const CleaningConfig& config, size_t chunk_size = 1024 * 1024);
    
    void process_stream(
        std::function<std::string()> input_reader,
        std::function<void(const std::string&)> output_writer
    );
    
    void process_file(const std::string& input_file, const std::string& output_file);
    
    const CleaningStats& get_stats() const { return stats_; }
    
private:
    CleaningConfig config_;
    size_t chunk_size_;
    CleaningStats stats_;
    std::unique_ptr<TextProcessor> processor_;
};

/**
 * Fuzzy string matching utilities for better deduplication.
 */
namespace fuzzy {
    
struct FuzzyMatch {
    double similarity;
    size_t edit_distance;
    std::vector<std::pair<size_t, size_t>> aligned_positions;
};

FuzzyMatch compute_fuzzy_similarity(const std::string& text1, const std::string& text2);
double jaccard_similarity(const std::string& text1, const std::string& text2);
double cosine_similarity(const std::string& text1, const std::string& text2);
size_t edit_distance(const std::string& text1, const std::string& text2);
std::vector<std::string> generate_ngrams(const std::string& text, size_t n);

} // namespace fuzzy

/**
 * Main text processing class with optimized implementations.
 * Thread-safe for concurrent usage.
 */
class TextProcessor {
public:
    explicit TextProcessor(const CleaningConfig& config = CleaningConfig{});
    ~TextProcessor() = default;
    
    // Copy and move constructors/operators
    TextProcessor(const TextProcessor& other) = delete;
    TextProcessor& operator=(const TextProcessor& other) = delete;
    TextProcessor(TextProcessor&& other) noexcept = default;
    TextProcessor& operator=(TextProcessor&& other) noexcept = default;
    
    /**
     * Clean and normalize text according to configuration.
     * 
     * @param text Input text to clean
     * @return Cleaned text
     */
    std::string clean_text(const std::string& text);
    
    /**
     * Normalize whitespace in text (fast path for common operation).
     * 
     * @param text Input text
     * @return Text with normalized whitespace
     */
    std::string normalize_whitespace(const std::string& text);
    
    /**
     * Remove stopwords from text efficiently.
     * 
     * @param text Input text
     * @param stopwords Set of stopwords to remove
     * @param preserve_semantic Whether to preserve semantic tokens
     * @return Text with stopwords removed
     */
    std::string strip_stopwords(
        const std::string& text, 
        const std::unordered_set<std::string>& stopwords,
        bool preserve_semantic = true
    );
    
    /**
     * Generate fast hash for sentence-level deduplication.
     * 
     * @param sentences Vector of sentences to hash
     * @return Vector of hash values (same order as input)
     */
    std::vector<uint64_t> hash_sentences(const std::vector<std::string>& sentences);
    
    /**
     * Deduplicate chunks based on similarity and exact matching.
     * 
     * @param chunks Input chunks
     * @param exact_threshold Threshold for exact hash matching
     * @param similarity_threshold Threshold for fuzzy matching (0.0-1.0)
     * @return Indices of chunks to keep
     */
    std::vector<size_t> dedup_chunks(
        const std::vector<std::string>& chunks,
        double exact_threshold = 0.95,
        double similarity_threshold = 0.85
    );
    
    /**
     * Get statistics from the last operation.
     * 
     * @return Processing statistics
     */
    const CleaningStats& get_last_stats() const { return last_stats_; }
    
    /**
     * Update configuration (thread-safe).
     * 
     * @param config New configuration
     */
    void update_config(const CleaningConfig& config);
    
    /**
     * Get current configuration.
     * 
     * @return Current configuration
     */
    CleaningConfig get_config() const { return config_; }

private:
    CleaningConfig config_;
    mutable CleaningStats last_stats_;
    
    // Internal optimization state
    struct Implementation;
    std::unique_ptr<Implementation> impl_;
    std::unique_ptr<MemoryPool> memory_pool_;
    std::unique_ptr<LanguageDetector> language_detector_;
    
    // Internal helper methods
    std::string normalize_unicode_internal(const std::string& text);
    std::string standardize_punctuation_internal(const std::string& text);
    std::string remove_control_chars_internal(const std::string& text);
    std::vector<std::string> split_sentences_internal(const std::string& text);
    bool is_semantic_token(const std::string& word);
    uint64_t compute_fast_hash(const std::string& text);
    double compute_similarity(const std::string& text1, const std::string& text2);
    
    // SIMD-optimized operations
    std::string normalize_whitespace_simd(const std::string& text);
    std::string remove_control_chars_simd(const std::string& text);
    
    // Parallel processing with work stealing
    std::vector<size_t> dedup_chunks_parallel(
        const std::vector<std::string>& chunks,
        double exact_threshold,
        double similarity_threshold
    );
};

// C-style API for maximum compatibility with Python bindings

extern "C" {
    
/**
 * Initialize the text processing library.
 * Call this once before using other functions.
 * 
 * @return 0 on success, non-zero on error
 */
int vecclean_init();

/**
 * Clean text with default configuration.
 * 
 * @param input Input text (null-terminated)
 * @param output Buffer for output text (allocated by caller)
 * @param output_size Size of output buffer
 * @return Length of cleaned text, or -1 on error
 */
int vecclean_clean_text_simple(const char* input, char* output, size_t output_size);

/**
 * Normalize whitespace in text.
 * 
 * @param input Input text (null-terminated)
 * @param output Buffer for output text (allocated by caller)
 * @param output_size Size of output buffer
 * @return Length of normalized text, or -1 on error
 */
int vecclean_normalize_whitespace(const char* input, char* output, size_t output_size);

/**
 * Remove stopwords from text.
 * 
 * @param input Input text (null-terminated)
 * @param stopwords Array of stopword strings
 * @param stopword_count Number of stopwords
 * @param output Buffer for output text (allocated by caller)
 * @param output_size Size of output buffer
 * @return Length of filtered text, or -1 on error
 */
int vecclean_strip_stopwords(
    const char* input,
    const char* const* stopwords,
    size_t stopword_count,
    char* output,
    size_t output_size
);

/**
 * Generate hash for a single sentence.
 * 
 * @param sentence Input sentence (null-terminated)
 * @return 64-bit hash value
 */
uint64_t vecclean_hash_sentence(const char* sentence);

/**
 * Get SIMD capabilities of the current system.
 * 
 * @return Bitmask of supported SIMD features
 */
uint32_t vecclean_get_simd_capabilities();

/**
 * Cleanup library resources.
 * Call this when done using the library.
 */
void vecclean_cleanup();

}  // extern "C"

// Utility functions and constants

namespace util {

/**
 * Fast string hashing function optimized for text similarity.
 * Uses xxHash for speed and quality.
 */
uint64_t fast_hash(const std::string& text);

/**
 * Compute Jaccard similarity between two strings.
 * Optimized for short to medium length texts.
 */
double jaccard_similarity(const std::string& text1, const std::string& text2);

/**
 * Check if a string contains only whitespace.
 * SIMD-optimized where available.
 */
bool is_whitespace_only(const std::string& text);

/**
 * Unicode normalization forms.
 */
enum class UnicodeForm {
    NFC,   // Canonical Decomposition followed by Canonical Composition
    NFD,   // Canonical Decomposition
    NFKC,  // Compatibility Decomposition followed by Canonical Composition
    NFKD   // Compatibility Decomposition
};

/**
 * Normalize Unicode string to specified form.
 */
std::string normalize_unicode(const std::string& text, UnicodeForm form);

/**
 * SIMD utilities for string operations.
 */
namespace simd {

bool has_sse2_support();
bool has_avx2_support();
std::string normalize_whitespace_sse2(const std::string& text);
std::string normalize_whitespace_avx2(const std::string& text);
std::string remove_control_chars_sse2(const std::string& text);
std::string remove_control_chars_avx2(const std::string& text);

} // namespace simd

/**
 * Work stealing thread pool for parallel processing.
 */
class WorkStealingThreadPool {
public:
    explicit WorkStealingThreadPool(size_t num_threads = 0);
    ~WorkStealingThreadPool();
    
    template<typename F>
    auto submit(F&& f) -> std::future<std_compat::result_of_t<F>>;
    
    void wait_for_all();
    size_t get_thread_count() const;
    
private:
    struct Implementation;
    std::unique_ptr<Implementation> impl_;
};

} // namespace util

} // namespace vecclean 