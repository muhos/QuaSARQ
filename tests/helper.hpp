#pragma once

#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <set>

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