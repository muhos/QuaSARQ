#pragma once

#include <mutex>
#include <iostream>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <cstddef>
#include <cstring>
#include <utility>
#include <cuda_runtime.h>
#include "color.hpp"
#include "constants.hpp"
#include "error.hpp"

#if __cplusplus >= 202002L
  using charu8_t = char8_t;
#else
  using charu8_t = char;
#endif

#define STARTLEN 10
#define RULERLEN 92

#define LOGGPU(FMT, ...)            printf(FMT, ##__VA_ARGS__)
#define LOGGPUERROR(FMT, ...)       printf(CERROR "ERROR: " FMT CNORMAL, ##__VA_ARGS__)

enum LogStream {
    LOG_STREAM_OUT = 0,
    LOG_STREAM_ERR = 1
};

using LogSink = void (*)(int stream, const char* text, void* user);

class Logger {

private:
    Logger() = default;
    int         verbose = 0;
    LogSink     sink = nullptr;
    void*       sink_user = nullptr;
    bool        silent = false;
    std::mutex  mutex;

    static Logger& get() {
        static Logger instance;
        return instance;
    }

    static void sink_nolock(FILE* stream, const char* text) {
        Logger& logger = get();
        logger.sink(stream == stderr ? LOG_STREAM_ERR : LOG_STREAM_OUT, text, logger.sink_user);
    }

    static void emit_nolock(FILE* stream, const char* text) {
        Logger& logger = get();
        if (logger.silent) return;
        if (logger.sink != nullptr) sink_nolock(stream, text);
        else std::fputs(text, stream);
    }

    static void emitch_nolock(FILE* stream, char ch) {
        Logger& logger = get();
        if (logger.silent) return;
        if (logger.sink != nullptr) {
            const char text[2] = { ch, '\0' };
            sink_nolock(stream, text);
        }
        else std::fputc(ch, stream);
    }

    template<typename... Args>
    static void emitf_nolock(FILE* stream, const char* fmt, Args&&... args) {
        Logger& logger = get();
        if (logger.silent) return;
        if (logger.sink == nullptr) {
            if constexpr (sizeof...(Args) > 0)
                std::fprintf(stream, fmt, std::forward<Args>(args)...);
            else
                std::fputs(fmt, stream);
            return;
        }
        if constexpr (sizeof...(Args) > 0) {
            char stackbuf[2048];
            const int needed = std::snprintf(stackbuf, sizeof(stackbuf), fmt, args...);
            if (needed < 0) return;
            if (size_t(needed) < sizeof(stackbuf)) {
                sink_nolock(stream, stackbuf);
                return;
            }
            std::string heapbuf(size_t(needed) + 1, '\0');
            std::snprintf(&heapbuf[0], heapbuf.size(), fmt, args...);
            sink_nolock(stream, heapbuf.c_str());
        }
        else {
            sink_nolock(stream, fmt);
        }
    }

    static void flush_nolock(FILE* stream) {
        Logger& logger = get();
        if (logger.silent || logger.sink != nullptr) return;
        std::fflush(stream);
    }

    static void repch_nolock(const char& ch, size_t times, FILE* stream = stdout) {
        Logger& logger = get();
        if (logger.silent || !times) return;
        if (logger.sink == nullptr) {
            while (times && times--) std::fputc(ch, stream);
            return;
        }
        const std::string text(times, ch);
        sink_nolock(stream, text.c_str());
    }

public:

    static void set_level(int lvl) noexcept { get().verbose = lvl; }

    static int min_verbosity() noexcept { return get().verbose; }

    static void set_sink(LogSink fn, void* user = nullptr) noexcept {
        Logger& logger = get();
        std::lock_guard<std::mutex> lock(logger.mutex);
        logger.sink = fn;
        logger.sink_user = user;
    }

    static void set_silent(bool on) noexcept {
        Logger& logger = get();
        std::lock_guard<std::mutex> lock(logger.mutex);
        logger.silent = on;
    }

    static bool is_silent() noexcept { return get().silent; }

    template<typename... Args>
    static void print(const char* fmt, FILE* stream, Args&&... args) {
        std::lock_guard<std::mutex> lock(get().mutex);
        emitf_nolock(stream, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void print(const char* fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(get().mutex);
        emitf_nolock(stdout, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void print(const charu8_t* fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(get().mutex);
        const char* c_fmt = reinterpret_cast<const char*>(fmt);
        emitf_nolock(stdout, c_fmt, std::forward<Args>(args)...);
    }

    static void putch(char ch) {
        std::lock_guard lock(get().mutex);
        emitch_nolock(stdout, ch);
    }

    static void repch(char ch, const size_t& times, FILE* stream = stdout) {
        std::lock_guard lock(get().mutex);
        repch_nolock(ch, times, stream);
    }

    static void ruler(int verbosity, char ch, size_t times) {
        if (min_verbosity() >= verbosity) {
            std::lock_guard lock(get().mutex);
            repch_nolock(ch, times);
            emitch_nolock(stdout, '\n');
        }
    }

    // Header logging
    static void header(int verbosity, int maxverbosity, const char* head, bool colored = true) {
        if (min_verbosity() >= verbosity && min_verbosity() < maxverbosity) {
            size_t len = std::strlen(head) + 4;  // brackets and spaces
            if (RULERLEN < len) {
                error("ruler length is smaller than header line (%zu)", len);
            }
            std::lock_guard lock(get().mutex);
            repch_nolock('-', STARTLEN);
            if (colored) {
                emitf_nolock(stdout, "[ %s%s%s ]", CHEADER, head, CNORMAL);
            }
            else {
                emitf_nolock(stdout, "[ %s ]", head);
            }
            repch_nolock('-', RULERLEN - len - STARTLEN);
            emitch_nolock(stdout, '\n');
        }
    }

    // Error logging
    template<typename... Args>
    static void error(const char* fmt, Args&&... args) {
        cudaDeviceSynchronize();
        char message[2048];
        if constexpr (sizeof...(Args) > 0)
            std::snprintf(message, sizeof(message), fmt, std::forward<Args>(args)...);
        else
            std::snprintf(message, sizeof(message), "%s", fmt);
        {
            std::lock_guard<std::mutex> lock(get().mutex);
            flush_nolock(stdout);
            emitf_nolock(stderr, "%sERROR: ", CERROR);
            emit_nolock(stderr, message);
            emitf_nolock(stderr, "\n%s", CNORMAL);
            flush_nolock(stderr);
        }
        throw QuaSARQ::fatal_error();
    }

    template<typename... Args>
    static void errorN(const char* fmt, Args&&... args) {
        std::lock_guard<std::mutex> lock(get().mutex);
        flush_nolock(stdout);
        emitf_nolock(stderr, "%s", CERROR);
        emitf_nolock(stderr, fmt, std::forward<Args>(args)...);
        emitf_nolock(stderr, "\n%s", CNORMAL);
        flush_nolock(stderr);
    }

    // Warning logging
    template<typename... Args>
    static void warning(const char* fmt, Args&&... args) {
        if (min_verbosity() >= 0) {
            std::lock_guard<std::mutex> lock(get().mutex);
            flush_nolock(stdout);
            emitf_nolock(stderr, "%sWARNING: ", CWARNING);
            emitf_nolock(stderr, fmt, std::forward<Args>(args)...);
            emitf_nolock(stderr, "\n%s", CNORMAL);
            flush_nolock(stderr);
        }
    }

    // Simple logs
    static void log0(const char* msg) {
        std::lock_guard<std::mutex> lock(get().mutex);
        emitf_nolock(stdout, "%s\n", msg);
    }

    static void logN0(const char* msg) {
        std::lock_guard<std::mutex> lock(get().mutex);
        emitf_nolock(stdout, "%s", msg);
    }

    template<typename... Args>
    static void log1(const char* fmt, Args&&... args) {
        if (min_verbosity() >= 1) {
            std::lock_guard<std::mutex> lock(get().mutex);
            emitf_nolock(stdout, fmt, std::forward<Args>(args)...);
            emitch_nolock(stdout, '\n');
        }
    }

    template<typename... Args>
    static void logN1(const char* fmt, Args&&... args) {
        if (min_verbosity() >= 1) {
            std::lock_guard<std::mutex> lock(get().mutex);
            emitf_nolock(stdout, fmt, std::forward<Args>(args)...);
        }
    }

    template<typename... Args>
    static void log2(int verbosity, const char* fmt, Args&&... args) {
        if (min_verbosity() >= verbosity) {
            std::lock_guard<std::mutex> lock(get().mutex);
            emitf_nolock(stdout, fmt, std::forward<Args>(args)...);
            emitch_nolock(stdout, '\n');
        }
    }

    template<typename... Args>
    static void logN2(int verbosity, const char* fmt, Args&&... args) {
        if (min_verbosity() >= verbosity) {
            std::lock_guard<std::mutex> lock(get().mutex);
            emitf_nolock(stdout, fmt, std::forward<Args>(args)...);
        }
    }

    // Done and ending
    static void done(int verbosity, int maxverbosity) {
        if (min_verbosity() >= verbosity && min_verbosity() < maxverbosity) {
            std::lock_guard<std::mutex> lock(get().mutex);
            emit_nolock(stdout, "done.\n");
        }
    }

    template<typename... Args>
    static void ending(int verbosity, int maxverbosity, const char* fmt, Args&&... args) {
        if (min_verbosity() >= verbosity && min_verbosity() < maxverbosity) {
            std::lock_guard<std::mutex> lock(get().mutex);
            emitf_nolock(stdout, fmt, std::forward<Args>(args)...);
            emit_nolock(stdout, " done.\n");
        }
    }

};

#define SET_LOGGER_VERBOSITY(V)         Logger::set_level(V)
#define SET_LOGGER_SINK(FN, USER)       Logger::set_sink(FN, USER)
#define SET_LOGGER_SILENT(ON)           Logger::set_silent(ON)
#define PRINT(FMT, ...)                 Logger::print(FMT, ##__VA_ARGS__)
#define PRINTFILE(FMT, FILE, ...)       Logger::print(FMT, FILE, ##__VA_ARGS__)
#define PUTCH(CH)                       Logger::putch(CH)
#define REPCH(CH, TIMES, ...)           Logger::repch(CH, TIMES, ##__VA_ARGS__)
#define LOGRULER(V, CH, TIMES)          Logger::ruler(V, CH, TIMES)
#define LOGHEADER(V, MV, HEAD)          Logger::header(V, MV, HEAD)
#define LOGHEADERNC(V, MV, HEAD)        Logger::header(V, MV, HEAD, false)
#define LOGERROR(FMT, ...)              Logger::error(FMT, ##__VA_ARGS__)
#define LOGERRORN(FMT, ...)             Logger::errorN(FMT, ##__VA_ARGS__)
#define LOGWARNING(FMT, ...)            Logger::warning(FMT, ##__VA_ARGS__)
#define LOG0(MSG)                       Logger::log0(MSG)
#define LOGN0(MSG)                      Logger::logN0(MSG)
#define LOG1(FMT, ...)                  Logger::log1(FMT, ##__VA_ARGS__)
#define LOGN1(FMT, ...)                 Logger::logN1(FMT, ##__VA_ARGS__)
#define LOG2(V, FMT, ...)               Logger::log2(V, FMT, ##__VA_ARGS__)
#define LOGN2(V, FMT, ...)              Logger::logN2(V, FMT, ##__VA_ARGS__)
#define LOGDONE(V, MV)                  Logger::done(V, MV)
#define LOGENDING(V, MV, FMT, ...)      Logger::ending(V, MV, FMT, ##__VA_ARGS__)
#define LOGPASSED(V)                    Logger::log2(V, "%sPASSED.%s", CGREEN, CNORMAL)
