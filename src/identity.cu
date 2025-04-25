
#include "simulator.hpp"
#include "identity.cuh"
#include "collapse.cuh"
#include "identitycheck.cuh"

namespace QuaSARQ {

    __global__ 
    void identity_1D(IDENTITY_ARGS) {
        for_parallel_x(q, num_qubits) {
            xs->set_word_to_identity(q, column_offset);
            zs->set_word_to_identity(q, column_offset);
        }
    }

    __global__ 
    void identity_Z_1D(IDENTITY_ARGS) {
        for_parallel_x(q, num_qubits) {
            zs->set_word_to_identity(q, column_offset);
        }
    }

    __global__ 
    void identity_X_1D(IDENTITY_ARGS) {
        for_parallel_x(q, num_qubits) {
            xs->set_word_to_identity(q, column_offset);
        }
    }

    __global__ 
    void identity_extended_1D(IDENTITY_ARGS) {
        for_parallel_x(q, num_qubits) {
            xs->set_stab_to_identity(q, column_offset);
            zs->set_stab_to_identity(q, column_offset);
        }
    }

    __global__ 
    void identity_Z_extended_1D(IDENTITY_ARGS) {
        for_parallel_x(q, num_qubits) {
            xs->set_destab_to_identity(q, column_offset);
            zs->set_stab_to_identity(q, column_offset);
        }
    }

    __global__ 
    void identity_X_extended_1D(IDENTITY_ARGS) {
        for_parallel_x(q, num_qubits) {
            zs->set_destab_to_identity(q, column_offset);
            xs->set_stab_to_identity(q, column_offset);
        }
    }
    
    void Simulator::identity(
                Tableau&        tab, 
        const   size_t&         offset_per_partition, 
        const   size_t&         num_qubits_per_partition, 
        const   cudaStream_t*   streams, 
        const   InitialState&   istate) 
    {
        const cudaStream_t& stream = streams[KERNEL_STREAM];
        if (options.tune_identity) {
            tune_identity(
                measuring ? identity_Z_extended_1D : identity_Z_1D, 
                bestblockidentity, 
                bestgrididentity, 
                offset_per_partition, 
                num_qubits_per_partition, 
                XZ_TABLE(tab));
        }
        char state = '0';
        if (istate == Plus)
            state = '+';
        else if (istate == Imag)
            state = 'i';
        LOGN2(1, "Creating \'%c\' initial state  for size %zd and offset %zd using grid(%d) and block(%d).. ", 
            state, num_qubits_per_partition, offset_per_partition, bestgrididentity.x, bestblockidentity.x);
        if (options.sync) cutimer.start();
        if (offset_per_partition) tab.reset();
        if (measuring) { 
            if (istate == Zero)
                identity_Z_extended_1D <<< bestgrididentity, bestblockidentity, 0, stream >>> 
                    (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
            else if (istate == Plus)
                identity_X_extended_1D <<< bestgrididentity, bestblockidentity, 0, stream >>> 
                    (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
            else if (istate == Imag)
                identity_extended_1D <<< bestgrididentity, bestblockidentity, 0, stream >>> 
                    (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
        }
        else {
            if (istate == Zero)
                identity_Z_1D <<< bestgrididentity, bestblockidentity, 0, stream >>> 
                    (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
            else if (istate == Plus)
                identity_X_1D <<< bestgrididentity, bestblockidentity, 0, stream >>> 
                    (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
            else if (istate == Imag)
                identity_1D <<< bestgrididentity, bestblockidentity, 0, stream >>> 
                    (offset_per_partition, num_qubits_per_partition, XZ_TABLE(tab));
        }
        if (options.sync) {
            LASTERR("failed to launch identity kernel");
            CHECK(cudaDeviceSynchronize());
            cutimer.stop();
            double itime = cutimer.time();
            LOG2(1, "done in %f ms.", itime);
        }
        else LOGDONE(1, 3);
        if (options.check_identity) {
            LOGN2(1, " Checking identity.. ");
            if (!check_identity(tableau, offset_per_partition, num_qubits_per_partition, measuring)) {
                LOGERROR("creating identity failed.");
            }
            LOG2(1, "%sPASSED.%s", CGREEN, CNORMAL);
        }
        if (options.print_initialtableau) 
            print_tableau(tab, MAX_DEPTH, false);
    }

}