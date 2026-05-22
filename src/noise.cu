#include "noise.cuh"

namespace QuaSARQ {

    __global__
    void setup_noise_k(
        curand_algorithm_t*         noise_states,
        const uint64                seed,
        const size_t                max_gates)
    {
        for_parallel_x(i, max_gates) {
            curand_init(seed, i, 0, &noise_states[i]);
        }
    }

    __global__
    void sample_noise_k(
        curand_algorithm_t*         noise_states,
        uint32*                     noise_paulis,
        const_refs_t                refs,
        const_buckets_t             gates,
        const size_t                num_gates)
    {
        for_parallel_x(i, num_gates) {
            const gate_ref_t ref  = refs[i];
            const Gate&      gate = (const Gate&) gates[ref];
            uint32 pauli = 0;
            curand_algorithm_t local = noise_states[i];
            const float prob = curand_uniform(&local);
            if (prob < gate.get_prob()) {
                pauli = gate.type == DEPOLARIZE1 ? 1u + (curand(&local) % 3u) :
                        gate.type == DEPOLARIZE2 ? 1u + (curand(&local) % 15u) :
                        gate.type == X_ERROR     ? 1u :
                        gate.type == Z_ERROR     ? 2u :
                        gate.type == Y_ERROR     ? 3u : 0u;
            }
            noise_states[i] = local; // advance sequence
            noise_paulis[i] = pauli;
        }
    }

}
