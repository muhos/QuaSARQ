#include "simulator.hpp"
#include "timer.hpp"

using namespace QuaSARQ;

#undef sign_update_X_or_Z
#undef sign_update_Y
#undef sign_update_H
#undef sign_update_S
#undef sign_update_Sdg
#undef sign_update_CX
#undef do_H
#undef do_S
#undef do_Sdg
#undef do_CX
#undef do_CZ
#undef do_CY
#undef do_CS
#undef do_SWAP

    #define sign_update_X_or_Z(SIGNS, SOURCE) SIGNS[w] ^= word_std_t(SOURCE[w])
    #define sign_update_Y(SIGNS, X, Z) SIGNS[w] ^= word_std_t((X[w] ^ Z[w]))
    #define sign_update_H(SIGNS, X, Z) SIGNS[w] ^= word_std_t((X[w] & Z[w]))
    #define sign_update_S(SIGNS, X, Z) sign_update_H(SIGNS, X, Z)
    #define sign_update_Sdg(SIGNS, X, Z) SIGNS[w] ^= word_std_t((X[w] & ~Z[w]))
    #define sign_update_CX(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc[w], xt = Xt[w], zc = Zc[w], zt = Zt[w]; \
        word_std_t xc_and_zt = xc & zt; \
        word_std_t not_xt_xor_zc = ~(xt ^ zc); \
        SIGNS[w] ^= word_std_t(xc_and_zt & not_xt_xor_zc); \
    }
    #define sign_update_CZ(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc[w], xt = Xt[w], zc = Zc[w], zt = Zt[w]; \
        word_std_t zc_xor_zt = zc ^ zt; \
        word_std_t xc_and_xt = xc & xt; \
        SIGNS[w] ^= word_std_t(zc_xor_zt & xc_and_xt); \
    }
    #define sign_update_CY(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc[w], xt = Xt[w], zc = Zc[w], zt = Zt[w]; \
        word_std_t zc_xor_zt = zc ^ zt; \
        word_std_t xc_and_xt = xc & xt; \
        SIGNS[w] ^= word_std_t(zc_xor_zt & xc_and_xt); \
    }
    #define sign_update_CS(SIGNS, Xc, Xt, Zc, Zt) \
    { \
        const word_std_t xc = Xc[w], xt = Xt[w], zc = Zc[w], zt = Zt[w]; \
        word_std_t left = (xc & zc) & (xt & ~zt); \
        word_std_t right = (xc & ~zc) & (xt & zt); \
        SIGNS[w] ^= word_std_t(left | right); \
    }

    #define do_H(SIGNS, EXT) \
    { \
        sign_update_H(SIGNS, x_## EXT, z_ ## EXT); \
        do_SWAP(z_ ## EXT[w], x_ ## EXT[w]); \
    }

    #define do_S(SIGNS, EXT) \
    { \
        sign_update_S(SIGNS, x_## EXT, z_ ## EXT); \
        z_ ## EXT[w].bitwise_xor(x_## EXT[w]); \
    }

    #define do_Sdg(SIGNS, EXT) \
    { \
        sign_update_Sdg(SIGNS, x_## EXT, z_ ## EXT); \
        z_ ## EXT[w].bitwise_xor(x_## EXT[w]); \
    }

    #define do_CX(SIGNS, C, T) \
    { \
        sign_update_CX(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C[w].bitwise_xor(z_words_ ## T[w]); \
        x_words_ ## T[w].bitwise_xor(x_words_ ## C[w]); \
    }

    #define do_CZ(SIGNS, C, T) \
    { \
        sign_update_CZ(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C[w].bitwise_xor(x_words_ ## T[w]); \
        z_words_ ## T[w].bitwise_xor(x_words_ ## C[w]); \
    }

    #define do_CY(SIGNS, C, T) \
    { \
        sign_update_CY(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C[w].bitwise_xor(x_words_ ## T[w] ^ z_words_ ## T[w]); \
        z_words_ ## T[w].bitwise_xor(x_words_ ## C[w]); \
        x_words_ ## T[w].bitwise_xor(x_words_ ## C[w]); \
    }

    #define do_CS(SIGNS, C, T) \
    { \
        sign_update_CS(SIGNS, x_words_ ## C, x_words_ ## T, z_words_ ## C, z_words_ ## T); \
        z_words_ ## C[w].bitwise_xor(x_words_ ## T[w]); \
        z_words_ ## T[w].bitwise_xor(x_words_ ## C[w]); \
    }

    #define do_SWAP(A,B) \
    { \
        word_t tmp = A; A = B; B = tmp; \
    }

inline void host_Z(const qubit_t& q, Table& xs, Signs& ss) {
    word_t* x_words = xs.words(q);
    sign_t* signs = ss.data(q);

    const size_t num_words_major = xs.num_words_major();

    for (size_t w = 0; w < num_words_major; w++) {

        sign_update_X_or_Z(signs, x_words);   

    }
}

inline void host_X(const qubit_t& q, Table& zs, Signs& ss) {
    word_t* z_words = zs.words(q);
    sign_t* signs = ss.data(q);

    const size_t num_words_major = zs.num_words_major();

    for (size_t w = 0; w < num_words_major; w++) {

        sign_update_X_or_Z(signs, z_words);
        
    }
}

inline void host_Y(const qubit_t& q, Table& xs, Table& zs, Signs& ss) {
    word_t* x_words = xs.words(q);
    word_t* z_words = zs.words(q);
    sign_t* signs = ss.data(q);

    const size_t num_words_major = xs.num_words_major();

    assert(num_words_major == zs.num_words_major());
    
    for (size_t w = 0; w < num_words_major; w++) {

        sign_update_Y(signs, x_words, z_words);

    }
}

inline void host_Hadamard(const qubit_t& q, Table& xs, Table& zs, Signs& ss) {
    // swap x with z
    word_t* x_words = xs.words(q);
    word_t* z_words = zs.words(q);
    sign_t* signs = ss.data(q);

    const size_t num_words_major = zs.num_words_major();

    assert(num_words_major == xs.num_words_major());

    for (size_t w = 0; w < num_words_major; w++) {

        do_H(signs, words);

    }
}

inline void host_Phase(const qubit_t& q, Table& xs, Table& zs, Signs& ss) {

    word_t* x_words = xs.words(q);
    word_t* z_words = zs.words(q);
    sign_t* signs = ss.data(q);

    const size_t num_words_major = zs.num_words_major();

    assert(num_words_major == xs.num_words_major());

    for (size_t w = 0; w < num_words_major; w++) {

        do_S(signs, words);

    }
}

inline void host_PhaseAdj(const qubit_t& q, Table& xs, Table& zs, Signs& ss) {

    word_t* x_words = xs.words(q);
    word_t* z_words = zs.words(q);
    sign_t* signs = ss.data(q);

    const size_t num_words_major = zs.num_words_major();

    assert(num_words_major == xs.num_words_major());

    for (size_t w = 0; w < num_words_major; w++) {

        do_Sdg(signs, words);

    }
}

inline void host_CX(const qubit_t& c, const qubit_t& t, Table& xs, Table& zs, Signs& ss) {
    word_t* x_words_c = xs.words(c);
    word_t* x_words_t = xs.words(t);
    word_t* z_words_c = zs.words(c);
    word_t* z_words_t = zs.words(t);
    sign_t* signs = ss.data(t);

    const size_t num_words_major = zs.num_words_major();

    assert(num_words_major == xs.num_words_major());

    for (size_t w = 0; w < num_words_major; w++) {

        do_CX(signs, c, t);
                    
    }
    
}

inline void host_CZ(const qubit_t& c, const qubit_t& t, Table& xs, Table& zs, Signs& ss) {
    word_t* x_words_c = xs.words(c);
    word_t* x_words_t = xs.words(t);
    word_t* z_words_c = zs.words(c);
    word_t* z_words_t = zs.words(t);
    sign_t* signs = ss.data(t);

    const size_t num_words_major = zs.num_words_major();

    assert(num_words_major == xs.num_words_major());

    for (size_t w = 0; w < num_words_major; w++) {

        do_CZ(signs, c, t);
                    
    }
    
}

inline void host_CY(const qubit_t& c, const qubit_t& t, Table& xs, Table& zs, Signs& ss) {
    word_t* x_words_c = xs.words(c);
    word_t* x_words_t = xs.words(t);
    word_t* z_words_c = zs.words(c);
    word_t* z_words_t = zs.words(t);
    sign_t* signs = ss.data(t);

    const size_t num_words_major = zs.num_words_major();

    assert(num_words_major == xs.num_words_major());

    for (size_t w = 0; w < num_words_major; w++) {

        do_CY(signs, c, t);
                    
    }
    
}

inline void host_Swap(const qubit_t& q1, const qubit_t& q2, Table& xs, Table& zs, Signs& ss) {
    word_t* x_words_q1 = xs.words(q1);
    word_t* x_words_q2 = xs.words(q2);
    word_t* z_words_q1 = zs.words(q1);
    word_t* z_words_q2 = zs.words(q2);

    const size_t num_words_major = zs.num_words_major();

    assert(num_words_major == xs.num_words_major());

    for (size_t w = 0; w < num_words_major; w++) {
        do_SWAP(x_words_q1[w], x_words_q2[w]);
        do_SWAP(z_words_q1[w], z_words_q2[w]);
    }
}

inline void host_iSwap(const qubit_t& q1, const qubit_t& q2, Table& xs, Table& zs, Signs& ss) {
    word_t* x_words_q1 = xs.words(q1);
    word_t* x_words_q2 = xs.words(q2);
    word_t* z_words_q1 = zs.words(q1);
    word_t* z_words_q2 = zs.words(q2);
    sign_t* signs_q1 = ss.data(q1);
    sign_t* signs_q2 = ss.data(q2);

    const size_t num_words_major = zs.num_words_major();

    assert(num_words_major == xs.num_words_major());

    for (size_t w = 0; w < num_words_major; w++) {

        // Swap(q1, q2)
        do_SWAP(x_words_q1[w], x_words_q2[w]);
        do_SWAP(z_words_q1[w], z_words_q2[w]);

        // CZ(q1, q2)
        do_CZ(signs_q2, q1, q2);

        // S(q1), S(q2)
        do_S(signs_q1, words_q1);
        do_S(signs_q2, words_q2);
    }
}

void Simulator::step_cpu_version(const Window& window) {
    for(gate_ref_t i = 0; i < window.size(); i++) {
        const gate_ref_t r = window[i];
        assert(r != NO_REF);
        const Gate* gate = circuit.gateptr(r); 
        switch(gate->type) {
            case I: 
                break;
            case H:
                host_Hadamard(gate->wires[0], host_xs, host_zs, host_ss);
                break;
            case S:
                host_Phase(gate->wires[0], host_xs, host_zs, host_ss);
                break;
            case Z:
                host_Z(gate->wires[0], host_xs, host_ss);
                break;
            case X:
                host_X(gate->wires[0], host_zs, host_ss);
                break;
            case Y:
                host_Y(gate->wires[0], host_xs, host_zs, host_ss);
                break;
            case Sdg:
                host_PhaseAdj(gate->wires[0], host_xs, host_zs, host_ss);
                break;
            case Swap:
                host_Swap(gate->wires[0], gate->wires[1], host_xs, host_zs, host_ss);
                break;
            case iSwap:
                host_iSwap(gate->wires[0], gate->wires[1], host_xs, host_zs, host_ss);
                break;
            case CX:
                host_CX(gate->wires[0], gate->wires[1], host_xs, host_zs, host_ss);
                break;
            case CZ:
                host_CZ(gate->wires[0], gate->wires[1], host_xs, host_zs, host_ss);
                break;
            case CY:
                host_CY(gate->wires[0], gate->wires[1], host_xs, host_zs, host_ss);
                break;
            default:
                break;               
        }
    }
}

void Simulator::step_cpu_version(const depth_t& depth_level) {
    // Benchmark a CPU version of step_2D.
    size_t num_qubits_padded = get_num_padded_bits(num_qubits);
    size_t num_words = get_num_words(num_qubits_padded * num_qubits_padded);
    size_t num_words_major = get_num_words(num_qubits);
    host_xs.alloc_host(num_words, num_words_major);
    host_zs.alloc_host(num_words, num_words_major);
    host_ss.alloc_host(num_words_major);
    timer.start();
    for(qubit_t q = 0; q < num_qubits; q++) {
		host_zs.set_word_to_identity(q);
	}
    timer.stop();
    double itime = timer.time();
    double avgRuntime = 0;
    BENCHMARK_CPU(step_cpu_version, avgRuntime, 10, circuit[depth_level]);
	LOG2(1, "CPU time for step: %f ms", avgRuntime + itime);
    host_xs.destroy(), host_zs.destroy(), host_ss.destroy();
}