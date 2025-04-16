#ifndef __CU_MACROS_H
#define __CU_MACROS_H

#if defined(INTERLEAVE_WORDS)
#define INTERLEAVE_XZ
#define INTERLEAVE_COLS 1
#endif

#if !defined(WORD_SIZE_8) && !defined(WORD_SIZE_32) && !defined(WORD_SIZE_64)
#define WORD_SIZE_64
#endif

#define FULL_WARP 0xFFFFFFFF

#define DEBUG_STEP 0

#endif