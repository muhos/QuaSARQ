#pragma once

#include <mutex>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <cstddef>
#include <cuda_runtime.h>
#include "color.hpp"
#include "constants.hpp"

#if __cplusplus >= 202002L
  using charu8_t = char8_t;
#else
  using charu8_t = char;
#endif

#define STARTLEN 10
#define RULERLEN 92

#define LOGGPU(FMT, ...)            printf(FMT, ##__VA_ARGS__)
#define LOGGPUERROR(FMT, ...)       printf(CERROR "ERROR: " FMT CNORMAL, ##__VA_ARGS__)

class Logger {

private:
    Logger() = default;
    int         verbose = 0;
    std::mutex  mutex;

    static Logger& get() {
        static Logger instance;
        return instance;
    }

    static void repch_nolock(const char& ch, size_t times) {
        while(times && times--) std::fputc(ch, stdout);
    }

public:

    static void set_level(int lvl) noexcept { get().verbose = lvl; }

    static int min_verbosity() noexcept { return get().verbose; }

    template<typename... Args>
    static void print(const char* fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(get().mutex);
        if constexpr (sizeof...(Args) > 0) {
            std::fprintf(stdout, fmt, std::forward<Args>(args)...);
        } else {
            std::fputs(fmt, stdout);
        }
    }

    template<typename... Args>
    static void print(const charu8_t* fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(get().mutex);
        const char* c_fmt = reinterpret_cast<const char*>(fmt);
        if constexpr (sizeof...(Args) > 0) {
            std::fprintf(stdout, c_fmt, std::forward<Args>(args)...);
        } else {
            std::fputs(c_fmt, stdout);
        }
    }

    static void putch(char ch) {
        std::lock_guard lock(get().mutex);
        std::fputc(ch, stdout);
    }

    static void repch(char ch, const size_t& times) {
        std::lock_guard lock(get().mutex);
        repch_nolock(ch, times);
    }

    static void ruler(int verbosity, char ch, size_t times) {
        if (min_verbosity() >= verbosity) {
            std::lock_guard lock(get().mutex);
            repch_nolock(ch, times);
            std::fputc('\n', stdout);
        }
    }

    // Header logging
    static void header(int verbosity, int maxverbosity, const char* head) {
        if (min_verbosity() >= verbosity && min_verbosity() < maxverbosity) {
            std::lock_guard lock(get().mutex);
            size_t len = std::strlen(head) + 4;  // brackets and spaces
            if (RULERLEN < len) {
                error("ruler length is smaller than header line (%zu)", len);
            }
            repch_nolock('-', STARTLEN);
            std::fprintf(stdout, "[ %s%s%s ]", CHEADER, head, CNORMAL);
            repch_nolock('-', RULERLEN - len - STARTLEN);
            std::fputc('\n', stdout);
        }
    }

    // Error logging
    template<typename... Args>
    static void error(const char* fmt, Args&&... args) {
        cudaDeviceSynchronize();
        std::lock_guard<std::mutex> lock(get().mutex);
        std::fprintf(stderr, "%sERROR: ", CERROR);
        if constexpr (sizeof...(Args) > 0)
            std::fprintf(stderr, fmt, std::forward<Args>(args)...);
        else 
            std::fputs(fmt, stderr);
        std::fprintf(stderr, "\n%s", CNORMAL);
        std::exit(1);
    }

    template<typename... Args>
    static void errorN(const char* fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(get().mutex);
        std::fprintf(stderr, "%s", CERROR);
        if constexpr (sizeof...(Args) > 0)
            std::fprintf(stderr, fmt, std::forward<Args>(args)...);
        else
            std::fputs(fmt, stderr);
        std::fprintf(stderr, "%s", CNORMAL);
    }

    // Warning logging
    template<typename... Args>
    static void warning(const char* fmt, Args&&... args) {
        if (min_verbosity() >= 0) {
            std::lock_guard<std::mutex> lock(get().mutex);
            std::fprintf(stderr, "%sWARNING: ", CWARNING);
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stderr, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stderr);
            std::fprintf(stderr, "\n%s", CNORMAL);
        }
    }

    // Simple logs
    static void log0(const char* msg) {
        std::lock_guard<std::mutex> lock(get().mutex);
        std::fprintf(stdout, "%s\n", msg);
    }

    static void logN0(const char* msg) {
        std::lock_guard<std::mutex> lock(get().mutex);
        std::fprintf(stdout, "%s", msg);
    }

    template<typename... Args>
    static void log1(const char* fmt, Args&&... args) {
        if (min_verbosity() >= 1) {
            std::lock_guard<std::mutex> lock(get().mutex);
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stdout, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stdout);
            std::fputc('\n', stdout);
        }
    }

    template<typename... Args>
    static void logN1(const char* fmt, Args&&... args) {
        if (min_verbosity() >= 1) {
            std::lock_guard<std::mutex> lock(get().mutex);
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stdout, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stdout);
        }
    }

    template<typename... Args>
    static void log2(int verbosity, const char* fmt, Args&&... args) {
        if (min_verbosity() >= verbosity) {
            std::lock_guard<std::mutex> lock(get().mutex);
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stdout, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stdout);
            std::fputc('\n', stdout);
        }
    }

    template<typename... Args>
    static void logN2(int verbosity, const char* fmt, Args&&... args) {
        if (min_verbosity() >= verbosity) {
            std::lock_guard<std::mutex> lock(get().mutex);
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stdout, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stdout);
        }
    }

    // Done and ending
    static void done(int verbosity, int maxverbosity) {
        if (min_verbosity() >= verbosity && min_verbosity() < maxverbosity) {
            std::lock_guard<std::mutex> lock(get().mutex);
            std::fprintf(stdout, "done.\n");
        }
    }

    template<typename... Args>
    static void ending(int verbosity, int maxverbosity, const char* fmt, Args&&... args) {
        if (min_verbosity() >= verbosity && min_verbosity() < maxverbosity) {
            std::lock_guard<std::mutex> lock(get().mutex);
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stdout, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stdout);
            std::fprintf(stdout, " done.\n");
        }
    }

};

#define SET_LOGGER_VERBOSITY(V)     Logger::set_level(V)
#define PRINT(FMT, ...)             Logger::print(FMT, ##__VA_ARGS__)
#define PUTCH(CH)                   Logger::putch(CH)
#define REPCH(CH, TIMES)            Logger::repch(CH, TIMES)
#define LOGRULER(V, CH, TIMES)      Logger::ruler(V, CH, TIMES)
#define LOGHEADER(V, MV, HEAD)      Logger::header(V, MV, HEAD)
#define LOGERROR(FMT, ...)          Logger::error(FMT, ##__VA_ARGS__)
#define LOGERRORN(FMT, ...)         Logger::errorN(FMT, ##__VA_ARGS__)
#define LOGWARNING(FMT, ...)        Logger::warning(FMT, ##__VA_ARGS__)
#define LOG0(MSG)                   Logger::log0(MSG)
#define LOGN0(MSG)                  Logger::logN0(MSG)
#define LOG1(FMT, ...)              Logger::log1(FMT, ##__VA_ARGS__)
#define LOGN1(FMT, ...)             Logger::logN1(FMT, ##__VA_ARGS__)
#define LOG2(V, FMT, ...)           Logger::log2(V, FMT, ##__VA_ARGS__)
#define LOGN2(V, FMT, ...)          Logger::logN2(V, FMT, ##__VA_ARGS__)
#define LOGDONE(V, MV)              Logger::done(V, MV)
#define LOGENDING(V, MV, FMT, ...)  Logger::ending(V, MV, FMT, ##__VA_ARGS__)
