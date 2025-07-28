/**
 * Pybind11 bindings for VecClean C++ text processing library.
 * 
 * This file creates Python bindings for the high-performance C++ implementations,
 * allowing seamless integration with the Python codebase while maintaining
 * maximum performance for CPU-intensive operations with advanced features.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

#include "cleaner.hpp"

// Include implementation to avoid incomplete type issues
#include "cleaner.cpp"

namespace py = pybind11;
using namespace vecclean;

// Helper functions for Python integration
py::dict memory_stats_to_dict(size_t used, size_t total, size_t allocations) {
    py::dict stats;
    stats["used_memory"] = used;
    stats["total_memory"] = total;
    stats["allocations"] = allocations;
    stats["utilization"] = total > 0 ? (double)used / total : 0.0;
    return stats;
}

py::dict profile_to_dict(const vecclean::PerformanceProfiler::ProfileEntry& entry) {
    py::dict profile;
    profile["function_name"] = entry.function_name;
    profile["duration_ms"] = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        entry.end_time - entry.start_time
    ).count();
    profile["memory_before"] = entry.memory_before;
    profile["memory_after"] = entry.memory_after;
    profile["memory_delta"] = entry.memory_after - entry.memory_before;
    
    py::dict metadata;
    for (const auto& [key, value] : entry.metadata) {
        metadata[key.c_str()] = value;
    }
    profile["metadata"] = metadata;
    
    return profile;
}

// Performance benchmarking function
py::dict benchmark_performance(const std::vector<std::string>& test_texts, int iterations = 1000) {
    auto start = std::chrono::high_resolution_clock::now();
    
    vecclean::TextProcessor processor;
    std::vector<std::string> results;
    results.reserve(test_texts.size());
    
    // Enable profiling for benchmarks
    vecclean::PerformanceProfiler::instance().set_enabled(true);
    
    for (int i = 0; i < iterations; ++i) {
        for (const auto& text : test_texts) {
            results.push_back(processor.clean_text(text));
        }
        if (i % 100 == 0) {
            results.clear();  // Prevent memory buildup
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Collect performance data
    auto profiles = vecclean::PerformanceProfiler::instance().get_profile_data();
    vecclean::PerformanceProfiler::instance().clear_profile_data();
    vecclean::PerformanceProfiler::instance().set_enabled(false);
    
    py::dict result;
    result["total_time_ms"] = duration.count();
    result["iterations"] = iterations;
    result["texts_per_iteration"] = test_texts.size();
    result["total_operations"] = iterations * test_texts.size();
    result["operations_per_second"] = (iterations * test_texts.size() * 1000.0) / duration.count();
    
    py::list profile_list;
    for (const auto& profile : profiles) {
        profile_list.append(profile_to_dict(profile));
    }
    result["profiles"] = profile_list;
    
    return result;
}

PYBIND11_MODULE(vecclean_cpp, m) {
    m.doc() = "VecClean C++ text processing library with high-performance implementations";
    
    // Module version and info
    m.attr("__version__") = "1.0.0";
    
    // Configuration struct
    py::class_<CleaningConfig>(m, "CleaningConfig", "Configuration for text cleaning operations")
        .def(py::init<>())
        .def_readwrite("normalize_unicode", &CleaningConfig::normalize_unicode)
        .def_readwrite("unicode_form", &CleaningConfig::unicode_form)
        .def_readwrite("normalize_whitespace", &CleaningConfig::normalize_whitespace)
        .def_readwrite("remove_extra_newlines", &CleaningConfig::remove_extra_newlines)
        .def_readwrite("trim_lines", &CleaningConfig::trim_lines)
        .def_readwrite("standardize_punctuation", &CleaningConfig::standardize_punctuation)
        .def_readwrite("standardize_quotes", &CleaningConfig::standardize_quotes)
        .def_readwrite("standardize_dashes", &CleaningConfig::standardize_dashes)
        .def_readwrite("remove_control_chars", &CleaningConfig::remove_control_chars)
        .def_readwrite("remove_empty_lines", &CleaningConfig::remove_empty_lines)
        .def_readwrite("min_line_length", &CleaningConfig::min_line_length)
        .def_readwrite("max_line_length", &CleaningConfig::max_line_length)
        .def_readwrite("min_text_length", &CleaningConfig::min_text_length)
        .def_readwrite("use_simd", &CleaningConfig::use_simd)
        .def_readwrite("parallel_processing", &CleaningConfig::parallel_processing)
        .def_readwrite("thread_count", &CleaningConfig::thread_count)
        .def_readwrite("enable_language_detection", &CleaningConfig::enable_language_detection)
        .def_readwrite("custom_replacements", &CleaningConfig::custom_replacements)
        .def_readwrite("use_memory_pool", &CleaningConfig::use_memory_pool)
        .def_readwrite("memory_pool_size", &CleaningConfig::memory_pool_size);
    
    // Statistics struct
    py::class_<CleaningStats>(m, "CleaningStats", "Statistics from text cleaning operations")
        .def(py::init<>())
        .def_readwrite("input_length", &CleaningStats::input_length)
        .def_readwrite("input_lines", &CleaningStats::input_lines)
        .def_readwrite("output_length", &CleaningStats::output_length)
        .def_readwrite("output_lines", &CleaningStats::output_lines)
        .def_readwrite("chars_normalized", &CleaningStats::chars_normalized)
        .def_readwrite("whitespace_normalized", &CleaningStats::whitespace_normalized)
        .def_readwrite("punctuation_standardized", &CleaningStats::punctuation_standardized)
        .def_readwrite("lines_removed", &CleaningStats::lines_removed)
        .def_readwrite("control_chars_removed", &CleaningStats::control_chars_removed)
        .def_readwrite("processing_time_ms", &CleaningStats::processing_time_ms)
        .def_readwrite("throughput_mb_per_sec", &CleaningStats::throughput_mb_per_sec)
        .def_readwrite("compression_ratio", &CleaningStats::compression_ratio)
        .def_readwrite("peak_memory_usage", &CleaningStats::peak_memory_usage)
        .def_readwrite("memory_allocations", &CleaningStats::memory_allocations)
        .def("reset", &CleaningStats::reset);
    
    // Main TextProcessor class
    py::class_<TextProcessor>(m, "TextProcessor", "High-performance text processor with C++ optimizations")
        .def(py::init<>())
        .def(py::init<const CleaningConfig&>())
        .def("clean_text", [](TextProcessor& processor, const std::string& text) {
            return processor.clean_text(text);
        }, "Clean and normalize text according to configuration", py::arg("text"))
        .def("normalize_whitespace", [](TextProcessor& processor, const std::string& text) {
            return processor.normalize_whitespace(text);
        }, "Normalize whitespace in text", py::arg("text"))
        .def("strip_stopwords", [](TextProcessor& processor, const std::string& text, 
                                   const std::unordered_set<std::string>& stopwords, bool preserve_semantic) {
            return processor.strip_stopwords(text, stopwords, preserve_semantic);
        }, "Remove stopwords from text", py::arg("text"), py::arg("stopwords"), py::arg("preserve_semantic") = true)
        .def("hash_sentences", &TextProcessor::hash_sentences, "Generate hashes for sentences", py::arg("sentences"))
        .def("dedup_chunks", &TextProcessor::dedup_chunks, "Deduplicate text chunks", 
             py::arg("chunks"), py::arg("exact_threshold") = 0.95, py::arg("similarity_threshold") = 0.85)
        .def("get_last_stats", &TextProcessor::get_last_stats, py::return_value_policy::reference_internal)
        .def("update_config", &TextProcessor::update_config, "Update processor configuration", py::arg("config"))
        .def("get_config", &TextProcessor::get_config, "Get current configuration")
        .def("__repr__", [](const TextProcessor& /* processor */) {
            return "<TextProcessor>";
        });
    
    // Utility functions
    m.def("get_capabilities", []() {
        py::dict caps;
        caps["simd_sse2"] = util::simd::has_sse2_support();
        caps["simd_avx2"] = util::simd::has_avx2_support();
        caps["version"] = "1.0.0";
        caps["build_type"] = "Release";
        return caps;
    }, "Get system capabilities and library info");
    
    // Performance functions
    m.def("enable_profiling", [](bool enabled) {
        vecclean::PerformanceProfiler::instance().set_enabled(enabled);
    }, "Enable or disable performance profiling", py::arg("enabled"));
    
    m.def("is_profiling_enabled", []() {
        return vecclean::PerformanceProfiler::instance().is_enabled();
    }, "Check if profiling is enabled");
    
    m.def("get_profile_data", []() {
        auto profiles = vecclean::PerformanceProfiler::instance().get_profile_data();
        py::list result;
        for (const auto& profile : profiles) {
            result.append(profile_to_dict(profile));
        }
        return result;
    }, "Get collected profiling data");
    
    m.def("clear_profile_data", []() {
        vecclean::PerformanceProfiler::instance().clear_profile_data();
    }, "Clear collected profiling data");
    
    m.def("benchmark_performance", &benchmark_performance, 
          "Benchmark text processing performance", 
          py::arg("test_texts"), py::arg("iterations") = 1000);
}

// Module initialization hook for debugging
#ifdef PYBIND11_MODULE_INIT_HOOK
PYBIND11_MODULE_INIT_HOOK(vecclean_cpp) {
    // Initialize C++ library
    vecclean_init();
    
#ifdef DEBUG
    // Enable profiling in debug builds
    vecclean::PerformanceProfiler::instance().set_enabled(true);
#endif
}
#endif 