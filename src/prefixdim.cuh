
#ifndef __CU_PREFIXDIM_H
#define __CU_PREFIXDIM_H

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


    #define POW2_Y_DIM_1(CALL) \
        case 1: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_1024(CALL, 1); \
            } \
            break;

    #define POW2_Y_DIM_2(CALL) \
        case 2: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_512(CALL, 2); \
            } \
            break;

    #define POW2_Y_DIM_4(CALL) \
        case 4: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_256(CALL, 4); \
            } \
            break;

    #define POW2_Y_DIM_8(CALL) \
        case 8: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_128(CALL, 8); \
            } \
            break;

    #define POW2_Y_DIM_16(CALL) \
        case 16: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_64(CALL, 16); \
            } \
            break;

    #define POW2_Y_DIM_32(CALL) \
        case 32: \
            switch (currentblock.x) { \
                FOREACH_X_DIM_MAX_32(CALL, 32); \
            } \
            break;
}

#endif