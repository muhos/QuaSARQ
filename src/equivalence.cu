#include "equivalence.hpp"
#include "step.cuh"
#include "collapse.cuh"
#include "operators.cuh"
#include "macros.cuh"
#include <cuda_profiler_api.h>

namespace QuaSARQ {

    __managed__ uint32 equivalent;

    #ifdef INTERLEAVE_XZ

    __global__ void equivalence_1D(const Table* ps, const Signs* ss, const Table* other_ps, const Signs* other_ss, const size_t min_num_words) {
        for_parallel_x(w, min_num_words) {
            if (equivalent && ((*ps)[w] != (*other_ps)[w])) {
                equivalent = 0;
                return;
            }
            else if (equivalent && ((*ps)[w] != (*other_ps)[w])) {
                equivalent = 0;
                return;
            }
            else if (equivalent && w < ss->size() && ((*ss)[w] != (*other_ss)[w])) {
                equivalent = 0;
                return;
            }
        }
    }

#else

    __global__ void equivalence_1D(const Table* xs, const Table* zs, const Signs* ss, const Table* other_xs, const Table* other_zs, const Signs* other_ss, const size_t min_num_words) {
        for_parallel_x(w, min_num_words) {
            if (equivalent && ((*xs)[w] != (*other_xs)[w])) {
                equivalent = 0;
                return;
            }
            else if (equivalent && ((*zs)[w] != (*other_zs)[w])) {
                equivalent = 0;
                return;
            }
            else if (equivalent && w < ss->size() && ((*ss)[w] != (*other_ss)[w])) {
                equivalent = 0;
                return;
            }
        }
    }

#endif


    bool Equivalence::check(const size_t& p, const cudaStream_t* streams, const cudaStream_t* other_streams) {

        double stime = 0;
        cudaStream_t copy_stream1 = streams[COPY_STREAM1];
        cudaStream_t copy_stream2 = streams[COPY_STREAM2];
        cudaStream_t kernel_stream = streams[KERNEL_STREAM];
        cudaStream_t other_copy_stream1 = other_streams[COPY_STREAM1];
        cudaStream_t other_copy_stream2 = other_streams[COPY_STREAM2];
        cudaStream_t other_kernel_stream = other_streams[KERNEL_STREAM];
        const size_t num_words_major = tableau.num_words_major();
        const size_t other_num_words_major = other_tableau.num_words_major();
        const size_t shared_element_bytes = sizeof(word_std_t);
        const size_t max_depth = MAX(depth, other_depth);

        if (options.disable_concurrency) {
            other_copy_stream1 = copy_stream1;
            other_copy_stream2 = copy_stream2;
            other_kernel_stream = kernel_stream;
            options.sync = true;
        }

        if (options.profile_equivalence)
            cudaProfilerStart();

        for (depth_t d = 0; d < max_depth; d++) {
            
            size_t num_gates_per_window = 0;
            size_t other_num_gates_per_window = 0;

            // Window transfer of first circuit.
            if (p < num_partitions && d < depth) {
                LOGN2(1, "Partition %zd: ", p);
                if (d > 0) SYNC(kernel_stream);
                gpu_circuit.copyfrom(stats, circuit, d, false, options.sync, copy_stream1, copy_stream2);
                num_gates_per_window = circuit[d].size();
                print_gates(gpu_circuit, num_gates_per_window, d);
            }

            // Window transfer of second circuit.
            if (p < other_num_partitions && d < other_depth) {
                LOGN2(1, "Partition %zd: ", p);
                if (d > 0) SYNC(other_kernel_stream);
                other_gpu_circuit.copyfrom(other_stats, other_circuit, d, false, options.sync, other_copy_stream1, other_copy_stream2);
                other_num_gates_per_window = other_circuit[d].size();
                print_gates(other_gpu_circuit, other_num_gates_per_window, d);
            }
            
#if DEBUG_STEP

            if (p < num_partitions && d < depth) {
                LOG1(" Debugging circuit-1 at depth %2d:", d);
                OPTIMIZESHARED(reduce_smem_size, bestBlockStep.y * bestBlockStep.x, shared_element_bytes);
                step_2D << < dim3(1, 1), dim3(1, 1), reduce_smem_size >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_major, XZ_TABLE(tableau), tableau.signs());
                LASTERR("failed to launch step kernel");
                SYNCALL;
            }

            if (p < other_num_partitions && d < other_depth) {
                LOG1(" Debugging circuit-2 at depth %2d:", d);
                OPTIMIZESHARED(reduce_smem_size, bestBlockStep.y * bestBlockStep.x, shared_element_bytes);
                step_2D << < dim3(1, 1), dim3(1, 1), reduce_smem_size >> > (other_gpu_circuit.references(), other_gpu_circuit.gates(), other_num_gates_per_window, other_num_words_major, XZ_TABLE(other_tableau), other_tableau.signs());
                LASTERR("failed to launch other step kernel");
                SYNCALL;             
            }

#else

            LOGN2(1, "Partition %zd: Checking equivalence the %d-time step %s using grid(%d, %d) and block(%d, %d).. ", 
                p, d, !options.sync ? "asynchronously" : "",
                bestGridStep.x, bestGridStep.y, bestBlockStep.x, bestBlockStep.y);

            if (options.sync) cutimer.start();

            OPTIMIZESHARED(reduce_smem_size, bestBlockStep.y * bestBlockStep.x, shared_element_bytes);

            if (p < num_partitions && d < depth) {
                SYNC(copy_stream1);
                SYNC(copy_stream2);
                if (bestBlockStep.x > maxWarpSize)
                    step_2D << < bestGridStep, bestBlockStep, reduce_smem_size, kernel_stream >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_major, XZ_TABLE(tableau), tableau.signs());
                else
                    step_2D_warped << < bestGridStep, bestBlockStep, reduce_smem_size, kernel_stream >> > (gpu_circuit.references(), gpu_circuit.gates(), num_gates_per_window, num_words_major, XZ_TABLE(tableau), tableau.signs());
            }

            if (p < other_num_partitions && d < other_depth) {
                SYNC(other_copy_stream1);
                SYNC(other_copy_stream2);
                if (bestBlockStep.x > maxWarpSize)
                    step_2D << < bestGridStep, bestBlockStep, reduce_smem_size, other_kernel_stream >> > (other_gpu_circuit.references(), other_gpu_circuit.gates(), other_num_gates_per_window, other_num_words_major, XZ_TABLE(other_tableau), other_tableau.signs());
                else
                    step_2D_warped << < bestGridStep, bestBlockStep, reduce_smem_size, other_kernel_stream >> > (other_gpu_circuit.references(), other_gpu_circuit.gates(), other_num_gates_per_window, other_num_words_major, XZ_TABLE(other_tableau), other_tableau.signs());
            }

            if (options.sync) { 
                LASTERR("failed to launch step kernel");
                cutimer.stop();
                stime = cutimer.time();
            }
            if (options.sync) {
                LOG2(1, "done in %f ms", stime);
            }
            else LOGDONE(1, 3);

#endif // End of debug/release mode.

            if (options.print_step_tableau && p < num_partitions && d < depth)
                print_tableau(tableau, d, false);
            if (options.print_step_tableau && p < other_num_partitions && d < other_depth)
                print_tableau(other_tableau, d, false);

        } // END of depth loop.

        // Check equivalence of two tableaus.
        SYNCALL;
        equivalent = 1;
        equivalence_1D << < bestGridIdentity, bestBlockIdentity >> > (XZ_TABLE(tableau), tableau.signs(), XZ_TABLE(other_tableau), other_tableau.signs(), MIN(tableau.num_words(), other_tableau.num_words()));
        LASTERR("failed to launch equivalence kernel");
        SYNC(0);
        if (equivalent)
            return 1;
        else
            return 0;

        if (options.profile_equivalence)
            cudaProfilerStop();

        if (options.disable_concurrency) {
            options.sync = false;
        }

    } // End of function.

}