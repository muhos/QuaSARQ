
#include "simulator.hpp"
#include "identity.cuh"
#include "collapse.cuh"
#include "tuner.cuh"

namespace QuaSARQ {

    // Set these to the tuned values (other than 1) 
    // to avoid trigeering the tuner. 
    dim3 bestBlockReset(96), bestGridReset(693);
    dim3 bestBlockIdentity(96), bestGridIdentity(693);

#ifdef INTERLEAVE_XZ

    __global__ void identity_1D(const size_t column_offset, const size_t num_qubits, Table* ps) {
        for_parallel_x(q, num_qubits) {
            ps->set_x_word_to_identity(q, column_offset);
            ps->set_z_word_to_identity(q, column_offset);
        }
    }

    __global__ void identity_Z_1D(const size_t column_offset, const size_t num_qubits, Table* ps) {
        for_parallel_x(q, num_qubits) {
            ps->set_z_word_to_identity(q, column_offset);
        }
    }

    __global__ void identity_X_1D(const size_t column_offset, const size_t num_qubits, Table* ps) {
        for_parallel_x(q, num_qubits) {
            ps->set_x_word_to_identity(q, column_offset);
        }
    }

#else

    __global__ void identity_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs) {
        for_parallel_x(q, num_qubits) {
            xs->set_word_to_identity(q, column_offset);
            zs->set_word_to_identity(q, column_offset);
        }
    }

    __global__ void identity_Z_1D(const size_t column_offset, const size_t num_qubits, Table* zs) {
        for_parallel_x(q, num_qubits) {
            zs->set_word_to_identity(q, column_offset);
        }
    }

    __global__ void identity_X_1D(const size_t column_offset, const size_t num_qubits, Table* xs) {
        for_parallel_x(q, num_qubits) {
            xs->set_word_to_identity(q, column_offset);
        }
    }

    __global__ void identity_extended_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs) {
        for_parallel_x(q, num_qubits) {
            xs->set_word_to_identity(q, column_offset, num_qubits);
            zs->set_word_to_identity(q, column_offset, num_qubits);
        }
    }

    __global__ void identity_Z_extended_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs) {
        for_parallel_x(q, num_qubits) {
            xs->set_word_to_identity(q, column_offset);
            zs->set_word_to_identity(q, column_offset, num_qubits);
        }
    }

    __global__ void identity_X_extended_1D(const size_t column_offset, const size_t num_qubits, Table* xs, Table* zs) {
        for_parallel_x(q, num_qubits) {
            zs->set_word_to_identity(q, column_offset);
            xs->set_word_to_identity(q, column_offset, num_qubits);
        }
    }

#endif
    
    void Simulator::identity(Tableau<DeviceAllocator>& tab, const size_t& offset_per_partition, const size_t& num_qubits_per_partition, const cudaStream_t* streams, const InitialState& istate) {
        assert(num_qubits_per_partition <= tab.num_qubits_padded());
        if (options.tune_identity) {
            tune_kernel(identity_Z_1D, "Identity", bestBlockIdentity, bestGridIdentity, offset_per_partition, num_qubits_per_partition, Z_TABLE(tab));
        }
        char state = '0';
        if (istate == Plus)
            state = '+';
        else if (istate == Imag)
            state = 'i';
        LOGN2(1, "Creating \'%c\' initial state  for size %zd and offset %zd using grid(%d) and block(%d).. ", state, num_qubits_per_partition, offset_per_partition, bestGridIdentity.x, bestBlockIdentity.x);
        if (options.sync) cutimer.start();
        if (offset_per_partition) tab.reset();
        if (measuring) { 
            if (istate == Zero)
                identity_Z_extended_1D <<< bestGridIdentity, bestBlockIdentity, 0, streams[KERNEL_STREAM] >>> (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
            else if (istate == Plus)
                identity_X_extended_1D <<< bestGridIdentity, bestBlockIdentity, 0, streams[KERNEL_STREAM] >>> (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
            else if (istate == Imag)
                identity_extended_1D <<< bestGridIdentity, bestBlockIdentity, 0, streams[KERNEL_STREAM] >>> (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
        }
        else {
            if (istate == Zero)
                identity_Z_1D <<< bestGridIdentity, bestBlockIdentity, 0, streams[KERNEL_STREAM] >>> (offset_per_partition, num_qubits_per_partition, Z_TABLE(tab));
            else if (istate == Plus)
                identity_X_1D <<< bestGridIdentity, bestBlockIdentity, 0, streams[KERNEL_STREAM] >>> (offset_per_partition, num_qubits_per_partition, X_TABLE(tab));
            else if (istate == Imag)
                identity_1D <<< bestGridIdentity, bestBlockIdentity, 0, streams[KERNEL_STREAM] >>> (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
        }
        if (options.sync) {
            LASTERR("failed to launch identity kernel");
            cutimer.stop();
            double itime = cutimer.time();
            LOG2(1, "done in %f ms.", itime);
        }
        else LOGDONE(1, 3);
        if (options.print_initial_tableau) 
            print_tableau(tab, -1, false);
    }

}