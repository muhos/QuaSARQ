#ifndef __SIMULATOR_H
#define __SIMULATOR_H

#include "vector.cuh"
#include "tableau.cuh"
#include "circuit.cuh"
#include "timer.cuh"
#include "timer.hpp"
#include "vector.hpp"
#include "random.hpp"
#include "circuit.hpp"
#include "parser.hpp"
#include "control.hpp"
#include "options.hpp"
#include "statistics.hpp"
#include "kernelconfig.hpp"


namespace QuaSARQ {

    class Simulator {

    protected:

        size_t                          num_qubits;
        size_t                          num_partitions;
        depth_t                         depth;
        Random                          random;
        Circuit                         circuit;
        CircuitIO                       circuit_io;
        string                          circuit_path;
        byte_t                          circuit_mode;
        Vec<qubit_t, size_t>            shuffled;
        Vec<byte_t, size_t>             locked;
        DeviceAllocator                 gpu_allocator;
        Tableau<DeviceAllocator>        tableau;
        DeviceCircuit<DeviceAllocator>  gpu_circuit;
        Table                           host_xs, host_zs;
        Signs                           host_ss;
        Statistics                      stats;
        FILE*                           configfile;
        cudaStream_t*                   custreams;
        
        enum { 
            COPY_STREAM1 = 0,
            COPY_STREAM2 = 1,
            KERNEL_STREAM = 2 
        };

        void register_config();
        bool open_config(arg_t file_mode = "r");
        void close_config();
        void create_streams(cudaStream_t*& streams);

    public:

       ~Simulator();
        Simulator();
        Simulator(const string& path);

        // Random circuit generation.
        void get_rand_qubit(const qubit_t& control, qubit_t& qubit);
        void shuffle_qubits();
        void generate();

        void initialize();
        void report();
        void parse();
        void simulate();
        size_t parse(Statistics& stats, const char* path);
        size_t schedule(Statistics& stats, Circuit& circuit);
        void simulate(const size_t& p, const bool& reversed);

        // Launch a kernel to make identity tableau.
        void identity(Tableau<DeviceAllocator>& tab, const size_t& offset_per_partition, const size_t& num_qubits_per_partition, const cudaStream_t* streams, const InitialState& istate = Zero);

        // Launch a kernel to update the tableau using
        // a single window of gates. Called step as it
        // advances the simulation by 1-time step.
        void step(const size_t& p, const depth_t& depth_level, const cudaStream_t* streams, const bool& reversed = false);

        // This will be moved to the checker later.
        void step_cpu_version(const depth_t& depth_level = 0);
        void step_cpu_version(const Window& window);

        // Printers.
        void print_tableau(const Tableau<DeviceAllocator>& tab, const depth_t& depth_level, const bool& reverse);
        void print_paulis(const Tableau<DeviceAllocator>& tab, const depth_t& depth_level, const bool& reversed);
        void print_gates(const DeviceCircuit<DeviceAllocator>& gates, const gate_ref_t& num_gates, const depth_t& depth_level);

    };

}

#endif