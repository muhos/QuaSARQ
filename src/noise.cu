#include "noise.cuh"

namespace QuaSARQ {

    __constant__ 
    constexpr uint32 PC2_PAULI[15] = {4, 12, 8, 1, 5, 13, 9, 3, 7, 15, 11, 2, 6, 14, 10};

    __global__
    void sample_noise_k(
        uint32*                     noise_paulis,
        const_refs_t                refs,
        const_buckets_t             gates,
        const size_t                num_gates,
        const uint64                seed,
        const uint32                step)
    {
        for_parallel_x(i, num_gates) {
            const gate_ref_t ref  = refs[i];
            const Gate&      gate = (const Gate&) gates[ref];
            uint32 pauli = 0;
            if (!noiseProbs(int(gate.type))) {
                noise_paulis[i] = pauli;
                continue;
            }
            const uint4 r = counter_bits(seed, RAND_TAG_NOISE << 16, uint32(i), step, 0);
            const float prob = float(r.x) * 2.3283064365386963e-10f;
            if (gate.type == PAULI_CHANNEL_1) {
                const float px = gate.get_prob(0), py = gate.get_prob(1), pz = gate.get_prob(2);
                pauli = prob < px ? 1u : prob < px + py ? 3u : prob < px + py + pz ? 2u : 0u;
            } 
            else if (gate.type == PAULI_CHANNEL_2) {
                float acc = 0.0f;
                for (uint32 k = 0; k < 15u; k++) {
                    acc += gate.get_prob(k);
                    if (prob < acc) { pauli = PC2_PAULI[k]; break; }
                }
            } 
            else if (prob < gate.get_prob(0)) {
                pauli = gate.type == DEPOLARIZE1 ? 1u + (r.y % 3u) :
                        gate.type == DEPOLARIZE2 ? 1u + (r.y % 15u) :
                        gate.type == X_ERROR     ? 1u :
                        gate.type == Z_ERROR     ? 2u :
                        gate.type == Y_ERROR     ? 3u :
                        gate.type == M           ? 1u :
                        gate.type == MR          ? 1u : 0u;
            }
            noise_paulis[i] = pauli;
        }
    }

}
