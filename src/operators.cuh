#pragma once

#include "table.cuh"
#include "gate.cuh"
#include "signs.cuh"
#include "atomic.cuh"

namespace QuaSARQ {

    /**********************************/
    /*** Clifford sign instructions ***/
    /**********************************/
    #define sign_update_global(SIGNS, VALUE) SIGNS ^= (VALUE)
    
    #define sign_update_X_or_Z(SIGNS, SOURCE) sign_update_global(SIGNS, word_std_t(SOURCE))

    #define sign_update_Y(SIGNS, X, Z) sign_update_global(SIGNS, word_std_t(X) ^ word_std_t(Z))

    #define sign_update_H(SIGNS, X, Z) sign_update_global(SIGNS, word_std_t(X) & word_std_t(Z))

    #define sign_update_S(SIGNS, X, Z) sign_update_H(SIGNS, X, Z)

    #define sign_update_Sdg(SIGNS, X, Z) sign_update_global(SIGNS, word_std_t(X) & ~word_std_t(Z))

    #define sign_update_CX(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc, xt = Xt, zc = Zc, zt = Zt; \
        const word_std_t not_xt_xor_zc = ~(xt ^ zc); \
        const word_std_t anding = xc & zt & not_xt_xor_zc; \
        sign_update_global(SIGNS, anding); \
    }

    #define sign_update_CZ(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc, xt = Xt, zc = Zc, zt = Zt; \
        const word_std_t zc_xor_zt = zc ^ zt; \
        const word_std_t anding = zc_xor_zt & xc & xt; \
        sign_update_global(SIGNS, anding); \
    }

    #define sign_update_CY(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc, xt = Xt, zc = Zc, zt = Zt; \
        const word_std_t xt_xor_zc = xt ^ zc; \
        const word_std_t xt_xor_zt = xt ^ zt; \
        const word_std_t anding = xc & xt_xor_zc & xt_xor_zt; \
        sign_update_global(SIGNS, anding); \
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
        z_ ## EXT ^= (x_## EXT); \
    }

    #define do_Sdg(SIGNS, EXT) \
    { \
        sign_update_Sdg(SIGNS, x_## EXT, z_ ## EXT); \
        z_ ## EXT ^= (x_## EXT); \
    }

    #define do_CX(SIGNS, C, T) \
    { \
        sign_update_CX(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C ^= (z_words_ ## T); \
        x_words_ ## T ^= (x_words_ ## C); \
    }

    #define do_CZ(SIGNS, C, T) \
    { \
        sign_update_CZ(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C ^= (x_words_ ## T); \
        z_words_ ## T ^= (x_words_ ## C); \
    }

    #define do_CY(SIGNS, C, T) \
    { \
        sign_update_CY(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C ^= (x_words_ ## T ^ z_words_ ## T); \
        z_words_ ## T ^= (x_words_ ## C); \
        x_words_ ## T ^= (x_words_ ## C); \
    }

    #define do_CS(SIGNS, C, T) \
    { \
        sign_update_CS(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C ^= (x_words_ ## T); \
        z_words_ ## T ^= (x_words_ ## C); \
    }

    #define do_SWAP(A,B) \
    { \
        word_t tmp = A; A = B; B = tmp; \
    }

    #define do_iSWAP(SIGNS, C, T) \
    { \
        /* SWAP(C, T) */ \
        do_SWAP(x_words_ ## C, x_words_ ## T); \
        do_SWAP(z_words_ ## C, z_words_ ## T); \
        /* CZ(C, T) */ \
        do_CZ(SIGNS, C, T); \
        /* S(C), S(T) */ \
        do_S(SIGNS, words_ ## C); \
        do_S(SIGNS, words_ ## T); \
    }
        
}