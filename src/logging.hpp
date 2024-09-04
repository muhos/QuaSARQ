
#ifndef __LOGGING_H
#define __LOGGING_H

#include <cstdio>
#include <cstring>
#include <string>
#include "constants.hpp"
#include "options.hpp"
#include "color.hpp"

#if defined(__linux__) || defined(__CYGWIN__)
#pragma GCC system_header
#endif

#define STARTLEN    10
#define RULELEN     92
#define PREFIX      ""
#define UNDERLINE	"\u001b[4m"

#define PUTCH(CH, ...) putc(CH, stdout)

#define PRINT(FORMAT, ...) fprintf(stdout, FORMAT, ## __VA_ARGS__)

#define LOGGPU(FORMAT, ...) printf(FORMAT, ## __VA_ARGS__)

#define LOGGPUERROR(FORMAT, ...) printf(CERROR "ERROR: " FORMAT CNORMAL, ## __VA_ARGS__)

#define LOGRULER(CH, TIMES) \
  do { \
     PUTCH(PREFIX[0]); \
     REPCH(CH, TIMES);      \
     PUTCH('\n'); \
  } while (0)

#define LOGERROR(FORMAT, ...) \
  do { \
     fprintf(stderr, CERROR "ERROR: " FORMAT "\n" CNORMAL, ## __VA_ARGS__); \
     exit(1); \
  } while (0)

#define LOGERRORN(FORMAT, ...) \
  do { \
     fprintf(stderr, CERROR "ERROR: " FORMAT "\n" CNORMAL, ## __VA_ARGS__); \
  } while (0)

#define LOGWARNING(FORMAT, ...) \
  do { \
     fprintf(stderr, CWARNING "WARNING: " FORMAT "\n" CNORMAL, ## __VA_ARGS__);\
  } while (0)

inline void REPCH(const char& ch, const size_t& size, const size_t& off = 0) {
    for (size_t i = off; i < size; i++) PUTCH(ch);
}

#define LOGHEADER(VERBOSITY, MAXVERBOSITY, HEAD) \
  do { \
    if (options.verbose >= VERBOSITY && options.verbose < MAXVERBOSITY) { \
	  size_t len = strlen(HEAD) + 4; \
	  if (RULELEN < len) LOGERROR("ruler length is smaller than header line (%zd)", len); \
      SETCOLOR(CNORMAL, stdout); \
	  REPCH('-', STARTLEN); \
	  PRINT("[ %s%s%s ]", CREPORT, HEAD, CNORMAL); \
	  REPCH('-', (RULELEN - len - STARTLEN)); \
	  PUTCH('\n'); \
    } \
  } while (0)

#define LOG0(MESSAGE) do { PRINT(PREFIX "%s\n", MESSAGE); } while (0)

#define LOGN0(MESSAGE) do { PRINT(PREFIX "%s", MESSAGE); } while (0)

#define LOG1(FORMAT, ...) \
    do { \
        PRINT(PREFIX FORMAT, ## __VA_ARGS__); PUTCH('\n'); \
    } while (0)

#define LOGN1(FORMAT, ...) \
    do { \
        PRINT(PREFIX FORMAT, ## __VA_ARGS__); \
    } while (0)

#define LOG2(VERBOSITY, FORMAT, ...) \
    do { \
        if (options.verbose >= VERBOSITY) { PRINT(PREFIX FORMAT, ## __VA_ARGS__); PUTCH('\n'); } \
    } while (0)

#define LOGN2(VERBOSITY, FORMAT, ...) \
    do { \
        if (options.verbose >= VERBOSITY) { PRINT(PREFIX FORMAT, ## __VA_ARGS__); } \
    } while(0)

#define LOGDONE(VERBOSITY, MAXVERBOSITY)  \
  do { \
    if (options.verbose >= VERBOSITY && options.verbose < MAXVERBOSITY) \
      PRINT("done.\n"); \
  } while(0)

#define LOGENDING(VERBOSITY, MAXVERBOSITY, FORMAT, ...) \
    do { \
        if (options.verbose >= VERBOSITY && options.verbose < MAXVERBOSITY) { \
            PRINT(FORMAT " done.\n", ## __VA_ARGS__); \
        } \
    } while(0)

#ifdef LOGGING

#TODO

#else // NO LOGGING

#define TODO_() do { } while (0)

#endif // NO LOGGING

#endif