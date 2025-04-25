#include "identitycheck.cuh"
#include "collapse.cuh"
#include "grid.cuh"
#include "options.hpp"
#include "access.cuh"

namespace QuaSARQ {

    __managed__ uint64 checksum;

    #define COLUMN_MAJOR_IDX(WORD_IDX, QUBIT_IDX) (QUBIT_IDX) * num_words_major + (WORD_IDX)

    template<int B>
    INLINE_DEVICE
    void identity_collapse(word_std_t xored, word_std_t* shared_xored, const size_t& num_qubits) {
        int tx = threadIdx.x;
        collapse_load_shared(shared_xored, xored, tx, num_qubits);
        collapse_shared<B, sign_t>(shared_xored, xored, tx);
        collapse_warp<B, sign_t>(xored, tx);
        if (!tx) {     
            atomicAdd(&checksum, __popcll(uint64(xored)));
        }
    }

    template<int B>
    __global__ 
    void check_identity_Z_2D(CHECK_IDENTITY_ARGS) {
        word_std_t* smem = SharedMemory<word_std_t>();
        word_std_t* shared_xored = smem + threadIdx.y * blockDim.x;
        for_parallel_y(w, num_words_major) {
            word_t xored_z = word_t(0), xored_x = word_t(0);
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !zs->check_word_is_identity(q, offset)) {
                    LOGGPUERROR("Z[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    zs->flag_not_indentity();                  
                }
                const size_t word_idx = COLUMN_MAJOR_IDX(w, q + offset);
                xored_z ^= (*zs)[word_idx], xored_x ^= (*xs)[word_idx];
            }
            word_std_t xored = xored_x ^ xored_z;
            identity_collapse<B>(xored, shared_xored, num_qubits);
        }
    }

    template<int B>
    __global__ 
    void check_identity_X_2D(CHECK_IDENTITY_ARGS) {
        word_std_t* smem = SharedMemory<word_std_t>();
        word_std_t* shared_xored = smem + threadIdx.y * blockDim.x;
        for_parallel_y(w, num_words_major) {
            word_t xored_z = word_t(0), xored_x = word_t(0);
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !xs->check_word_is_identity(q, offset)) {
                    LOGGPUERROR("X[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    xs->flag_not_indentity();
                }
                const size_t word_idx = COLUMN_MAJOR_IDX(w, q + offset);
                xored_z ^= (*zs)[word_idx], xored_x ^= (*xs)[word_idx];
            }
            word_std_t xored = xored_x ^ xored_z;
            identity_collapse<B>(xored, shared_xored, num_qubits);
        }
    }

    template<int B>
    __global__ 
    void check_identity_2D(CHECK_IDENTITY_ARGS) {
        word_std_t* smem = SharedMemory<word_std_t>();
        word_std_t* shared_xored = smem + threadIdx.y * blockDim.x;
        for_parallel_y(w, num_words_major) {
            word_t xored_z = word_t(0), xored_x = word_t(0);
            for_parallel_x(q, num_qubits) {
                if (!w && !zs->check_word_is_identity(q, offset)) {
                    LOGGPUERROR("Z[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    zs->flag_not_indentity();
                }
                if (!w && !xs->check_word_is_identity(q, offset)) {
                    LOGGPUERROR("X[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    xs->flag_not_indentity();
                }
                const size_t word_idx = COLUMN_MAJOR_IDX(w, q + offset);
                xored_z ^= (*zs)[word_idx], xored_x ^= (*xs)[word_idx];
            }
            word_std_t xored = xored_x ^ xored_z;
            identity_collapse<B>(xored, shared_xored, num_qubits);
        }
    }

    template<int B>
    __global__ 
    void check_identity_Z_extended_2D(CHECK_IDENTITY_ARGS) {
        word_std_t* smem = SharedMemory<word_std_t>();
        word_std_t* shared_xored = smem + threadIdx.y * blockDim.x;
        for_parallel_y(w, num_words_major) {
            word_t xored_z = word_t(0), xored_x = word_t(0);
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !zs->check_stab_is_identity(q, offset)) {
                    LOGGPUERROR("Z[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    zs->flag_not_indentity();                  
                }
                if (!w && !xs->check_destab_is_identity(q, offset)) {
                    LOGGPUERROR("X[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    xs->flag_not_indentity();
                }
                const size_t word_idx = COLUMN_MAJOR_IDX(w, q + offset);
                xored_z ^= (*zs)[word_idx], xored_x ^= (*xs)[word_idx];
            }
            word_std_t xored = xored_x ^ xored_z;
            identity_collapse<B>(xored, shared_xored, num_qubits);
        }
    }

    template<int B>
    __global__ 
    void check_identity_X_extended_2D(CHECK_IDENTITY_ARGS) {
        word_std_t* smem = SharedMemory<word_std_t>();
        word_std_t* shared_xored = smem + threadIdx.y * blockDim.x;
        for_parallel_y(w, num_words_major) {
            word_t xored_z = word_t(0), xored_x = word_t(0);
            for_parallel_x(q, num_qubits) {
                // Check the diagonal. Done only once.
                if (!w && !zs->check_destab_is_identity(q, offset)) {
                    LOGGPUERROR("Z[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    zs->flag_not_indentity();
                }
                if (!w && !xs->check_stab_is_identity(q, offset)) {
                    LOGGPUERROR("X[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    xs->flag_not_indentity();                  
                }
                const size_t word_idx = COLUMN_MAJOR_IDX(w, q + offset);
                xored_z ^= (*zs)[word_idx], xored_x ^= (*xs)[word_idx];
            }
            word_std_t xored = xored_x ^ xored_z;
            identity_collapse<B>(xored, shared_xored, num_qubits);
        }
    }

    template<int B>
    __global__ 
    void check_identity_extended_2D(CHECK_IDENTITY_ARGS) {
        word_std_t* smem = SharedMemory<word_std_t>();
        word_std_t* shared_xored = smem + threadIdx.y * blockDim.x;
        for_parallel_y(w, num_words_major) {
            word_t xored_z = word_t(0), xored_x = word_t(0);
            for_parallel_x(q, num_qubits) {
                if (!w && !zs->check_stab_is_identity(q, offset)) {
                    LOGGPUERROR("Z[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    zs->flag_not_indentity();
                }
                if (!w && !xs->check_stab_is_identity(q, offset)) {
                    LOGGPUERROR("X[w: %lld, q: %lld] is incorrect.\n", w, (q + offset));
                    xs->flag_not_indentity();
                }
                const size_t word_idx = COLUMN_MAJOR_IDX(w, q + offset);
                xored_z ^= (*zs)[word_idx], xored_x ^= (*xs)[word_idx];
            }
            word_std_t xored = xored_x ^ xored_z;
            identity_collapse<B>(xored, shared_xored, num_qubits);
        }
    }


    #define XBLOCKSIZE 32

    bool check_identity(
        const Tableau&  tableau, 
        const size_t&   offset_per_partition, 
        const size_t&   num_qubits_per_partition,
        const bool&     measuring) {
        assert(num_qubits_per_partition <= tableau.num_qubits_padded());
        SYNCALL;
        dim3 currentblock(XBLOCKSIZE, 16); 
        dim3 currentgrid(50, 50);
        OPTIMIZESHARED(reduce_smem_size, currentblock.y * currentblock.x, sizeof(word_std_t));
        checksum = 0;
        void (*kernel)(const size_t, const size_t, const size_t, Table*, Table*);
        if (measuring) { 
            if (options.initialstate == Zero) {
                kernel = check_identity_Z_extended_2D<XBLOCKSIZE>;
            }
            else if (options.initialstate == Plus) {
                kernel = check_identity_X_extended_2D<XBLOCKSIZE>;
            }
            else {
                assert(options.initialstate == Imag);
                kernel = check_identity_extended_2D<XBLOCKSIZE>;
            }
        }
        else {
            if (options.initialstate == Zero) {
                kernel = check_identity_Z_2D<XBLOCKSIZE>;
            }
            else if (options.initialstate == Plus) {
                kernel = check_identity_X_2D<XBLOCKSIZE>;
            }
            else {
                assert(options.initialstate == Imag);
                kernel = check_identity_2D<XBLOCKSIZE>;
            }
        }
        kernel <<< currentgrid, currentblock, reduce_smem_size >>> 
                (offset_per_partition, num_qubits_per_partition, tableau.num_words_major(), XZ_TABLE(tableau));
        LASTERR("failed to launch check-identity kernel");
        SYNCALL;
        if (options.initialstate == Imag)
            return tableau.is_table_identity() && checksum == 0;
        else
            return tableau.is_table_identity() && 
            checksum == measuring ? 2 * num_qubits_per_partition : num_qubits_per_partition;
    }

}