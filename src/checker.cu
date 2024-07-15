
#include "checker.hpp"
#include "identity.cuh"
#include "collapse.cuh"

namespace QuaSARQ {

    // Set these to the tuned values (other than 1) 
    // to avoid trigeering the tuner. 
    dim3 bestBlockCheckIdentity(2, 128), bestGridCheckIdentity(89, 17);

    __managed__ uint64 checksum;

    __global__ void check_identity_Z_2D(const size_t offset, const size_t num_qubits, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps
    #else
    Table* xs, Table* zs
    #endif
    ) {

        word_std_t* shared_xored = SharedMemory<word_std_t>();
        grid_t tx = threadIdx.x;
        grid_t bx = blockDim.x;
        grid_t global_offset = blockIdx.x * blockDim.x;
        grid_t collapse_tid = threadIdx.y * blockDim.x + tx;

        for_parallel_y(w, num_words_major) {

            word_std_t xored_z = word_std_t(0);
            word_std_t xored_x = word_std_t(0);

    #ifdef INTERLEAVE_XZ
            word_t* gens_per_word = ps->data() + w;
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !ps->check_z_word_is_identity(q, offset)) {
                    LOGGPU(" Qubit %lld\n", (q + offset));
                    ps->flag_not_indentity();                  
                }
                xored_z ^= word_std_t(gens_per_word[(Z_OFFSET(q) + offset) * num_words_major]);
                xored_x ^= word_std_t(gens_per_word[(X_OFFSET(q) + offset) * num_words_major]);
            }
    #else
            word_t* z_gens_per_word = zs->data() + w;
            word_t* x_gens_per_word = xs->data() + w;
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !zs->check_word_is_identity(q, offset)) {
                    LOGGPU(" Qubit %lld\n", (q + offset));
                    zs->flag_not_indentity();                  
                }
                xored_z ^= word_std_t(z_gens_per_word[(q + offset) * num_words_major]);
                xored_x ^= word_std_t(x_gens_per_word[(q + offset) * num_words_major]);
            }
    #endif

            word_std_t xored = xored_x ^ xored_z;

            load_shared(shared_xored, xored, collapse_tid, tx, num_qubits);

            collapse_shared(shared_xored, xored, collapse_tid, bx, tx);

            collapse_warp(shared_xored, xored, collapse_tid, bx, tx);

            if (!tx && global_offset < num_qubits) {     
                atomicAdd(&checksum, __popcll(uint64(xored)));
            }

        }
    }

    __global__ void check_identity_X_2D(const size_t offset, const size_t num_qubits, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps
    #else
    Table* xs, Table* zs
    #endif
    ) {

        word_std_t* shared_xored = SharedMemory<word_std_t>();
        grid_t tx = threadIdx.x;
        grid_t bx = blockDim.x;
        grid_t global_offset = blockIdx.x * bx;
        grid_t collapse_tid = threadIdx.y * bx + tx;

        for_parallel_y(w, num_words_major) {

            word_std_t xored_z = word_std_t(0);
            word_std_t xored_x = word_std_t(0);

    #ifdef INTERLEAVE_XZ
            word_t* gens_per_word = ps->data() + w;
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !ps->check_x_word_is_identity(q, offset)) {
                    LOGGPU(" Qubit %lld\n", (q + offset));
                    ps->flag_not_indentity();                  
                }
                xored_z ^= word_std_t(gens_per_word[(Z_OFFSET(q) + offset) * num_words_major]);
                xored_x ^= word_std_t(gens_per_word[(X_OFFSET(q) + offset) * num_words_major]);
            }
    #else
            word_t* z_gens_per_word = zs->data() + w;
            word_t* x_gens_per_word = xs->data() + w;
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !xs->check_word_is_identity(q, offset)) {
                    xs->flag_not_indentity();
                }
                xored_z ^= word_std_t(z_gens_per_word[(q + offset) * num_words_major]);
                xored_x ^= word_std_t(x_gens_per_word[(q + offset) * num_words_major]);
            }
    #endif

            word_std_t xored = xored_x ^ xored_z;

            load_shared(shared_xored, xored, collapse_tid, tx, num_qubits);

            collapse_shared(shared_xored, xored, collapse_tid, bx, tx);

            collapse_warp(shared_xored, xored, collapse_tid, bx, tx);

            if (!tx && global_offset < num_qubits) {
                atomicAdd(&checksum, __popcll(uint64(xored)));
            }

        }
    }

    __global__ void check_identity_2D(const size_t offset, const size_t num_qubits, const size_t num_words_major, 
    #ifdef INTERLEAVE_XZ
    Table* ps
    #else
    Table* xs, Table* zs
    #endif
    ) {

        word_std_t* shared_xored = SharedMemory<word_std_t>();
        grid_t tx = threadIdx.x;
        grid_t bx = blockDim.x;
        grid_t global_offset = blockIdx.x * blockDim.x;
        grid_t collapse_tid = threadIdx.y * blockDim.x + tx;

        for_parallel_y(w, num_words_major) {

            word_std_t xored_z = word_std_t(0);
            word_std_t xored_x = word_std_t(0);

    #ifdef INTERLEAVE_XZ
            word_t* gens_per_word = ps->data() + w;
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !ps->check_z_word_is_identity(q, offset)) {
                    ps->flag_not_indentity();
                }
                if (!w && !ps->check_x_word_is_identity(q, offset)) {
                    ps->flag_not_indentity();
                }
                xored_z ^= word_std_t(gens_per_word[(Z_OFFSET(q) + offset) * num_words_major]);
                xored_x ^= word_std_t(gens_per_word[(X_OFFSET(q) + offset) * num_words_major]);
            }
    #else
            word_t* z_gens_per_word = zs->data() + w;
            word_t* x_gens_per_word = xs->data() + w;
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !zs->check_word_is_identity(q, offset)) {
                    zs->flag_not_indentity();
                }
                if (!w && !xs->check_word_is_identity(q, offset)) {
                    xs->flag_not_indentity();
                }
                xored_z ^= word_std_t(z_gens_per_word[(q + offset) * num_words_major]);
                xored_x ^= word_std_t(x_gens_per_word[(q + offset) * num_words_major]);
            }
    #endif
            word_std_t xored = xored_x ^ xored_z;

            load_shared(shared_xored, xored, collapse_tid, tx, num_qubits);

            collapse_shared(shared_xored, xored, collapse_tid, bx, tx);

            collapse_warp(shared_xored, xored, collapse_tid, bx, tx);

            if (!tx && global_offset < num_qubits) {
                atomicAdd(&checksum, __popcll(uint64(xored)));
            }

        }
    }

    bool Checker::check_identity(const size_t& offset_per_partition, const size_t& num_qubits_per_partition) {
        assert(num_qubits_per_partition <= tableau.num_qubits_padded());
        SYNCALL;
        if (options.initialstate == Zero) {         
            OPTIMIZESHARED(reduce_smem_size, bestBlockCheckIdentity.y * bestBlockCheckIdentity.x, sizeof(word_std_t));
            checksum = 0;
            check_identity_Z_2D << < bestGridCheckIdentity, bestBlockCheckIdentity, reduce_smem_size >> > (offset_per_partition, num_qubits_per_partition, tableau.num_words_major(), XZ_TABLE(tableau));
            LASTERR("failed to launch identity kernel");
            // Call is_table_identity() is blocking, thus it's
            // safe to read checksum on host afterwards.
            return tableau.is_table_identity() && checksum == num_qubits_per_partition;
        }
        else if (options.initialstate == Plus) {
            OPTIMIZESHARED(reduce_smem_size, bestBlockCheckIdentity.y * bestBlockCheckIdentity.x, sizeof(word_std_t));
            checksum = 0;
            check_identity_X_2D << < bestGridCheckIdentity, bestBlockCheckIdentity, reduce_smem_size >> > (offset_per_partition, num_qubits_per_partition, tableau.num_words_major(), XZ_TABLE(tableau));
            LASTERR("failed to launch identity kernel");
            // Call is_table_identity() is blocking, thus it's
            // safe to read checksum on host afterwards.
            return tableau.is_table_identity() && checksum == num_qubits_per_partition;
        }
        else {
            assert(options.initialstate == Imag);
            OPTIMIZESHARED(reduce_smem_size, bestBlockCheckIdentity.y * bestBlockCheckIdentity.x, sizeof(word_std_t));
            checksum = 0;
            check_identity_2D << < bestGridCheckIdentity, bestBlockCheckIdentity, reduce_smem_size >> > (offset_per_partition, num_qubits_per_partition, tableau.num_words_major(), XZ_TABLE(tableau));
            LASTERR("failed to launch identity kernel");
            // Call is_table_identity() is blocking, thus it's
            // safe to read checksum on host afterwards.
            return tableau.is_table_identity() && checksum == 0;
        }

    }

}