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

    /**********************************/
    /*** Clifford gate instructions ***/
    /**********************************/

    #define update_SWAP(A,B) \
    { \
        word_t tmp = A; A = B; B = tmp; \
    }

    #define update_H(EXT) \
    { \
        update_SWAP(z_ ## EXT, x_ ## EXT); \
    }

    #define update_S(EXT) \
    { \
        z_ ## EXT ^= (x_## EXT); \
    }

    #define update_CX(C, T) \
    { \
        z_words_ ## C ^= (z_words_ ## T); \
        x_words_ ## T ^= (x_words_ ## C); \
    }

    #define update_CZ(C, T) \
    { \
        z_words_ ## C ^= (x_words_ ## T); \
        z_words_ ## T ^= (x_words_ ## C); \
    }

    #define update_CY(C, T) \
    { \
        z_words_ ## C ^= (x_words_ ## T ^ z_words_ ## T); \
        z_words_ ## T ^= (x_words_ ## C); \
        x_words_ ## T ^= (x_words_ ## C); \
    }

    #define update_iSWAP(C, T) \
    { \
        /* SWAP(C, T) */ \
        update_SWAP(x_words_ ## C, x_words_ ## T); \
        update_SWAP(z_words_ ## C, z_words_ ## T); \
        /* CZ(C, T) */ \
        update_CZ(C, T); \
        /* S(C), S(T) */ \
        update_S(words_ ## C); \
        update_S(words_ ## T); \
    }

    #define do_H(SIGNS, EXT) \
    { \
        sign_update_H(SIGNS, x_## EXT, z_ ## EXT); \
        update_SWAP(z_ ## EXT, x_ ## EXT); \
    }

    #define do_S(SIGNS, EXT) \
    { \
        sign_update_S(SIGNS, x_## EXT, z_ ## EXT); \
        update_S(EXT); \
    }

    #define do_Sdg(SIGNS, EXT) \
    { \
        sign_update_Sdg(SIGNS, x_## EXT, z_ ## EXT); \
        update_S(EXT); \
    }

    #define do_CX(SIGNS, C, T) \
    { \
        sign_update_CX(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        update_CX(C, T); \
    }

    #define do_CZ(SIGNS, C, T) \
    { \
        sign_update_CZ(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        update_CZ(C, T); \
    }

    #define do_CY(SIGNS, C, T) \
    { \
        sign_update_CY(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        update_CY(C, T); \
    }

    #define do_SWAP(A, B) \
    { \
        update_SWAP(A, B); \
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

    #define do_iSWAPdg(SIGNS, C, T) \
    { \
        do_Sdg(SIGNS, words_ ## T); \
        do_Sdg(SIGNS, words_ ## C); \
        do_CZ(SIGNS, C, T); \
        do_SWAP(z_words_ ## C, z_words_ ## T); \
        do_SWAP(x_words_ ## C, x_words_ ## T); \
    }
        
}