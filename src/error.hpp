#pragma once

#include <stdexcept>
#include <string>

namespace QuaSARQ {

    class fatal_error : public std::runtime_error {
    public:
        fatal_error() : std::runtime_error("fatal error") {}
        explicit fatal_error(const std::string& message) : std::runtime_error(message) {}
        explicit fatal_error(const char* message) : std::runtime_error(message) {}
    };

}
