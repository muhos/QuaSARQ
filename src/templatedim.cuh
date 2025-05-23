#pragma once

namespace QuaSARQ {
    
    #define POW2_X_DIM(CALL, XDIM, YDIM) \
        case XDIM: CALL(XDIM, YDIM); break; 

    #define FOREACH_X_DIM_MAX_1024(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) \
        POW2_X_DIM(CALL, 8, YDIM) \
        POW2_X_DIM(CALL, 16, YDIM) \
        POW2_X_DIM(CALL, 32, YDIM) \
        POW2_X_DIM(CALL, 64, YDIM) \
        POW2_X_DIM(CALL, 128, YDIM) \
        POW2_X_DIM(CALL, 256, YDIM) \
        POW2_X_DIM(CALL, 512, YDIM) \
        POW2_X_DIM(CALL, 1024, YDIM) 

    #define FOREACH_X_DIM_MAX_512(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) \
        POW2_X_DIM(CALL, 8, YDIM) \
        POW2_X_DIM(CALL, 16, YDIM) \
        POW2_X_DIM(CALL, 32, YDIM) \
        POW2_X_DIM(CALL, 64, YDIM) \
        POW2_X_DIM(CALL, 128, YDIM) \
        POW2_X_DIM(CALL, 256, YDIM) \
        POW2_X_DIM(CALL, 512, YDIM) 

    #define FOREACH_X_DIM_MAX_256(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) \
        POW2_X_DIM(CALL, 8, YDIM) \
        POW2_X_DIM(CALL, 16, YDIM) \
        POW2_X_DIM(CALL, 32, YDIM) \
        POW2_X_DIM(CALL, 64, YDIM) \
        POW2_X_DIM(CALL, 128, YDIM) \
        POW2_X_DIM(CALL, 256, YDIM) 

    #define FOREACH_X_DIM_MAX_128(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) \
        POW2_X_DIM(CALL, 8, YDIM) \
        POW2_X_DIM(CALL, 16, YDIM) \
        POW2_X_DIM(CALL, 32, YDIM) \
        POW2_X_DIM(CALL, 64, YDIM) \
        POW2_X_DIM(CALL, 128, YDIM) 

    #define FOREACH_X_DIM_MAX_64(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) \
        POW2_X_DIM(CALL, 8, YDIM) \
        POW2_X_DIM(CALL, 16, YDIM) \
        POW2_X_DIM(CALL, 32, YDIM) \
        POW2_X_DIM(CALL, 64, YDIM) 

    #define FOREACH_X_DIM_MAX_32(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) \
        POW2_X_DIM(CALL, 8, YDIM) \
        POW2_X_DIM(CALL, 16, YDIM) \
        POW2_X_DIM(CALL, 32, YDIM) 

    #define FOREACH_X_DIM_MAX_16(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) \
        POW2_X_DIM(CALL, 8, YDIM) \
        POW2_X_DIM(CALL, 16, YDIM) 

    #define FOREACH_X_DIM_MAX_8(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) \
        POW2_X_DIM(CALL, 8, YDIM) 

    #define FOREACH_X_DIM_MAX_4(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) \
        POW2_X_DIM(CALL, 4, YDIM) 

    #define FOREACH_X_DIM_MAX_2(CALL, YDIM) \
        POW2_X_DIM(CALL, 2, YDIM) 

    #define DEFAULT_CASE_X_DIM \
        default: LOGERROR("unknown x-block size (%lld) of prefix kernel", currentblock.x); break

    #define POW2_Y_DIM_1(CALL) \
        case 1: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_1024(CALL, 1); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_2(CALL) \
        case 2: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_512(CALL, 2); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_4(CALL) \
        case 4: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_256(CALL, 4); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_8(CALL) \
        case 8: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_128(CALL, 8); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_16(CALL) \
        case 16: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_64(CALL, 16); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_32(CALL) \
        case 32: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_32(CALL, 32); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_64(CALL) \
        case 64: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_16(CALL, 64); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_128(CALL) \
        case 128: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_8(CALL, 128); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_256(CALL) \
        case 256: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_4(CALL, 256); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define POW2_Y_DIM_512(CALL) \
        case 512: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_2(CALL, 512); \
                DEFAULT_CASE_X_DIM; \
            } \
            break;

    #define GENERATE_SWITCH_FOR_CALL(CALL) \
        switch (currentblock.y) { \
            POW2_Y_DIM_1(CALL); \
            POW2_Y_DIM_2(CALL); \
            POW2_Y_DIM_4(CALL); \
            POW2_Y_DIM_8(CALL); \
            POW2_Y_DIM_16(CALL); \
            POW2_Y_DIM_32(CALL); \
            POW2_Y_DIM_64(CALL); \
            POW2_Y_DIM_128(CALL); \
            POW2_Y_DIM_256(CALL); \
            POW2_Y_DIM_512(CALL); \
            default: \
                LOGERROR("unknown y-block size (%lld) of prefix kernel", currentblock.y); \
                break; \
        }
}