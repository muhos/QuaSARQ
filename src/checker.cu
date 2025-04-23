
#include "checker.hpp"
#include "identity.cuh"
#include "collapse.cuh"
#include "templatedim.cuh"

namespace QuaSARQ {

    __managed__ uint64 checksum;

    template<int B>
    __global__ 
    void check_identity_Z_2D(
        const   size_t offset, 
        const   size_t num_qubits, 
        const   size_t num_words_major, 
                Table* xs, 
                Table* zs
    ) {
        word_std_t* smem = SharedMemory<word_std_t>();
        int tx = threadIdx.x;
        grid_t BX = blockDim.x;
        sign_t* shared_xored = smem + threadIdx.y * BX;
        grid_t global_offset = blockIdx.x * blockDim.x;

        for_parallel_y(w, num_words_major) {

            word_std_t xored_z = word_std_t(0);
            word_std_t xored_x = word_std_t(0);

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

            word_std_t xored = xored_x ^ xored_z;

            collapse_load_shared(shared_xored, xored, tx, num_qubits);

            collapse_shared<B, sign_t>(shared_xored, xored, tx);

            collapse_warp<B, sign_t>(xored, tx);

            if (!tx && global_offset < num_qubits) {     
                atomicAdd(&checksum, __popcll(uint64(xored)));
            }

        }
    }

    template<int B>
    __global__ 
    void check_identity_X_2D(
        const   size_t offset, 
        const   size_t num_qubits, 
        const   size_t num_words_major, 
                Table* xs, 
                Table* zs
    ) {

        word_std_t* smem = SharedMemory<word_std_t>();
        int tx = threadIdx.x;
        grid_t BX = blockDim.x;
        sign_t* shared_xored = smem + threadIdx.y * BX;
        grid_t global_offset = blockIdx.x * blockDim.x;

        for_parallel_y(w, num_words_major) {

            word_std_t xored_z = word_std_t(0);
            word_std_t xored_x = word_std_t(0);

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

            word_std_t xored = xored_x ^ xored_z;

            collapse_load_shared(shared_xored, xored, tx, num_qubits);

            collapse_shared<B, sign_t>(shared_xored, xored, tx);

            collapse_warp<B, sign_t>(xored, tx);

            if (!tx && global_offset < num_qubits) {
                atomicAdd(&checksum, __popcll(uint64(xored)));
            }

        }
    }

    template<int B>
    __global__ 
    void check_identity_2D(
        const   size_t offset, 
        const   size_t num_qubits, 
        const   size_t num_words_major, 
                Table* xs, 
                Table* zs
    ) {

        word_std_t* smem = SharedMemory<word_std_t>();
        int tx = threadIdx.x;
        grid_t BX = blockDim.x;
        sign_t* shared_xored = smem + threadIdx.y * BX;
        grid_t global_offset = blockIdx.x * blockDim.x;

        for_parallel_y(w, num_words_major) {

            word_std_t xored_z = word_std_t(0);
            word_std_t xored_x = word_std_t(0);

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

            word_std_t xored = xored_x ^ xored_z;

            collapse_load_shared(shared_xored, xored, tx, num_qubits);

            collapse_shared<B, sign_t>(shared_xored, xored, tx);

            collapse_warp<B, sign_t>(xored, tx);

            if (!tx && global_offset < num_qubits) {
                atomicAdd(&checksum, __popcll(uint64(xored)));
            }

        }
    }

    #define XBLOCKSIZE 2

    bool Checker::check_identity(const size_t& offset_per_partition, const size_t& num_qubits_per_partition) {
        assert(num_qubits_per_partition <= tableau.num_qubits_padded());
        SYNCALL;
        dim3 bestBlockCheckIdentity(XBLOCKSIZE, 128); 
        dim3 bestGridCheckIdentity(89, 17);
        if (options.initialstate == Zero) {         
            OPTIMIZESHARED(reduce_smem_size, bestBlockCheckIdentity.y * bestBlockCheckIdentity.x, sizeof(word_std_t));
            checksum = 0;
            check_identity_Z_2D<XBLOCKSIZE> << < bestGridCheckIdentity, bestBlockCheckIdentity, reduce_smem_size >> > (offset_per_partition, num_qubits_per_partition, tableau.num_words_major(), XZ_TABLE(tableau));
            LASTERR("failed to launch identity kernel");
            // Call is_table_identity() is blocking, thus it's
            // safe to read checksum on host afterwards.
            return tableau.is_table_identity() && checksum == num_qubits_per_partition;
        }
        else if (options.initialstate == Plus) {
            OPTIMIZESHARED(reduce_smem_size, bestBlockCheckIdentity.y * bestBlockCheckIdentity.x, sizeof(word_std_t));
            checksum = 0;
            check_identity_X_2D<XBLOCKSIZE> << < bestGridCheckIdentity, bestBlockCheckIdentity, reduce_smem_size >> > (offset_per_partition, num_qubits_per_partition, tableau.num_words_major(), XZ_TABLE(tableau));
            LASTERR("failed to launch identity kernel");
            // Call is_table_identity() is blocking, thus it's
            // safe to read checksum on host afterwards.
            return tableau.is_table_identity() && checksum == num_qubits_per_partition;
        }
        else {
            assert(options.initialstate == Imag);
            OPTIMIZESHARED(reduce_smem_size, bestBlockCheckIdentity.y * bestBlockCheckIdentity.x, sizeof(word_std_t));
            checksum = 0;
            check_identity_2D<XBLOCKSIZE> << < bestGridCheckIdentity, bestBlockCheckIdentity, reduce_smem_size >> > (offset_per_partition, num_qubits_per_partition, tableau.num_words_major(), XZ_TABLE(tableau));
            LASTERR("failed to launch identity kernel");
            // Call is_table_identity() is blocking, thus it's
            // safe to read checksum on host afterwards.
            return tableau.is_table_identity() && checksum == 0;
        }

    }

}