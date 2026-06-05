#pragma once

#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <set>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <system_error>
#include <cctype>

#include "color.hpp"

struct TestFailure : std::runtime_error {
    explicit TestFailure(const char* expr, int line)
        : std::runtime_error(std::string("line ") + std::to_string(line) + ": " + expr) {}
};

#define TCHECK(expr) \
    if (!(expr)) throw TestFailure(#expr, __LINE__)

inline int total = 0, passed = 0;

inline void section(const char* title) {
    std::cout << std::format("\n{}{}{}\n", CHEADER, title, CNORMAL);
    std::cout << std::string(60, '-') << '\n';
}

template<typename Func>
void run_test(const char* name, Func func) {
    ++total;
    try {
        func();
        std::cout << std::format("  {}{}{} {}PASS{}\n", CTEST, name, CNORMAL, CPASS, CNORMAL);
        ++passed;
    }
    catch (const TestFailure& e) {
        std::cerr << std::format("  {}FAIL  {}  ({}){}\n", CFAIL, name, e.what(), CNORMAL);
    }
    catch (const std::exception& e) {
        std::cerr << std::format("  {}FAIL  {}  (unexpected: {}){}\n", CFAIL, name, e.what(), CNORMAL);
    }
}

inline std::vector<std::string> circuit_paths() {
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator("circuits")) {
        if (entry.is_regular_file() && entry.path().extension() == ".stim")
            paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

inline size_t circuit_distance(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    size_t pos = name.find("_d");
    while (pos != std::string::npos) {
        pos += 2;
        if (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
            size_t distance = 0;
            while (pos < name.size() && std::isdigit(static_cast<unsigned char>(name[pos]))) {
                distance = distance * 10 + size_t(name[pos] - '0');
                pos++;
            }
            return distance;
        }
        pos = name.find("_d", pos);
    }
    return 0;
}

inline std::vector<std::string> circuit_paths_up_to_distance(const size_t& max_distance) {
    std::vector<std::string> paths;
    for (const auto& path : circuit_paths()) {
        const size_t distance = circuit_distance(path);
        if (distance > 0 && distance <= max_distance)
            paths.push_back(path);
    }
    return paths;
}

inline void cleanup_generated_measure_files() {
    for (const auto& entry : std::filesystem::directory_iterator("circuits")) {
        if (entry.is_regular_file() && entry.path().extension() == ".01") {
            std::error_code ec;
            std::filesystem::remove(entry.path(), ec);
        }
    }
}
