#ifndef __CU_CLIFFORDUTILS_H
#define __CU_CLIFFORDUTILS_H

#include "table.cuh"
#include "gate.cuh"
#include "signs.cuh"

namespace QuaSARQ {

    /**********************************/
    /*** Clifford sign instructions ***/
    /**********************************/
#if defined(WORD_SIZE_8)
    #if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
    INLINE_DEVICE sign_t
    #else
    INLINE_DEVICE void
    #endif
    atomicXOR(sign_t* addr, const uint32& value) {
        assert(value <= WORD_MAX);
        uint64 addr_val = (uint64)addr;
        uint32* al_addr = reinterpret_cast<uint32*> (addr_val & (0xFFFFFFFFFFFFFFFCULL));
        uint32 al_offset = uint32(addr_val & 3) << 3;
        uint32 byte = value << al_offset;
        #if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
        return sign_t((atomicXor(al_addr, byte) >> al_offset) & 0xFF);
        #else
        atomicXor(al_addr, byte);
        #endif
    }
#else
    INLINE_DEVICE sign_t
    atomicXOR(sign_t* addr, const word_std_t& value) {
        return atomicXor(addr, value);
    }
#endif

    #define sign_update_global(SIGNS, VALUE) SIGNS ^= (VALUE)
    
    #define sign_update_X_or_Z(SIGNS, SOURCE) sign_update_global(SIGNS, word_std_t(SOURCE))

    #define sign_update_Y(SIGNS, X, Z) sign_update_global(SIGNS, word_std_t(X) ^ word_std_t(Z))

    #define sign_update_H(SIGNS, X, Z) sign_update_global(SIGNS, word_std_t(X) & word_std_t(Z))

    #define sign_update_S(SIGNS, X, Z) sign_update_H(SIGNS, X, Z)

    #define sign_update_Sdg(SIGNS, X, Z) sign_update_global(SIGNS, word_std_t(X) & ~word_std_t(Z))

    #define sign_update_CX(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc, xt = Xt, zc = Zc, zt = Zt; \
        const word_std_t xc_and_zt = xc & zt; \
        const word_std_t not_xt_xor_zc = ~(xt ^ zc); \
        sign_update_global(SIGNS, xc_and_zt & not_xt_xor_zc); \
    }

    #define sign_update_CZ(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc, xt = Xt, zc = Zc, zt = Zt; \
        const word_std_t zc_xor_zt = zc ^ zt; \
        const word_std_t xc_and_xt = xc & xt; \
        sign_update_global(SIGNS, zc_xor_zt & xc_and_xt); \
    }

    #define sign_update_CY(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc, xt = Xt, zc = Zc, zt = Zt; \
        const word_std_t xt_xor_zc = xt ^ zc; \
        const word_std_t xt_xor_zt = xt ^ zt; \
        sign_update_global(SIGNS, xc & (xt_xor_zc & xt_xor_zt)); \
    }

    #define sign_update_CS(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc, xt = Xt, zc = Zc, zt = Zt; \
        const word_std_t left = (xc & zc) & (xt & ~zt); \
        const word_std_t right = (xc & ~zc) & (xt & zt); \
        sign_update_global(SIGNS, left | right); \
    }

    /**********************************/
    /*** Clifford gate instructions ***/
    /**********************************/

    #define do_H(SIGNS, EXT) \
    { \
        sign_update_H(SIGNS, x_## EXT, z_ ## EXT); \
        do_SWAP(z_ ## EXT, x_ ## EXT); \
    }

    #define do_S(SIGNS, EXT) \
    { \
        sign_update_S(SIGNS, x_## EXT, z_ ## EXT); \
        z_ ## EXT.bitwise_xor(x_## EXT); \
    }

    #define do_Sdg(SIGNS, EXT) \
    { \
        sign_update_Sdg(SIGNS, x_## EXT, z_ ## EXT); \
        z_ ## EXT.bitwise_xor(x_## EXT); \
    }

    #define do_CX(SIGNS, C, T) \
    { \
        sign_update_CX(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C.bitwise_xor(z_words_ ## T); \
        x_words_ ## T.bitwise_xor(x_words_ ## C); \
    }

    #define do_CZ(SIGNS, C, T) \
    { \
        sign_update_CZ(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C.bitwise_xor(x_words_ ## T); \
        z_words_ ## T.bitwise_xor(x_words_ ## C); \
    }

    #define do_CY(SIGNS, C, T) \
    { \
        sign_update_CY(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C.bitwise_xor(x_words_ ## T ^ z_words_ ## T); \
        z_words_ ## T.bitwise_xor(x_words_ ## C); \
        x_words_ ## T.bitwise_xor(x_words_ ## C); \
    }

    #define do_CS(SIGNS, C, T) \
    { \
        sign_update_CS(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C.bitwise_xor(x_words_ ## T); \
        z_words_ ## T.bitwise_xor(x_words_ ## C); \
    }

    #define do_SWAP(A,B) \
    { \
        word_t tmp = A; A = B; B = tmp; \
    }

    #define do_iSWAP(SIGNS, C, T) \
    { \
        /* Swap(C, T) */ \
        do_SWAP(x_words_ ## C, x_words_ ## T); \
        do_SWAP(z_words_ ## C, z_words_ ## T); \
        /* CZ(C, T) */ \
        do_CZ(SIGNS, C, T); \
        /* S(C), S(T) */ \
        do_S(SIGNS, words_ ## C); \
        do_S(SIGNS, words_ ## T); \
    }
        
}

#endif