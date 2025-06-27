#pragma once

#include "locker.cuh"
#include "tableau.cuh"
#include "circuit.cuh"
#include "prefix.cuh"
#include "timer.hpp"
#include "vector.hpp"
#include "random.hpp"
#include "circuit.hpp"
#include "parser.hpp"
#include "statistics.hpp"


namespace QuaSARQ {

    class Simulator {

    protected:

        size_t                          num_qubits;
        size_t                          num_partitions;
        depth_t                         depth;
        Random                          crand;
        Random                          mrand;
        Circuit                         circuit;
        CircuitIO                       circuit_io;
        string                          circuit_path;
        byte_t                          circuit_mode;
        Vec<qubit_t, size_t>            measurements;
        Vec<qubit_t, size_t>            shuffled;
        Vec<byte_t, size_t>             locked;
        DeviceAllocator                 gpu_allocator;
        DeviceCircuit                   gpu_circuit;
        Locker                          locker;
        Tableau                         tableau;
        Tableau                         inv_tableau;
        Pivoting                        pivoting; 
        MeasurementChecker              mchecker;
        Prefix                          prefix;
        Statistics                      stats;
        Timer                           progress_timer;
        FILE*                           config_file;
        size_t                          config_qubits;
        cudaStream_t*                   custreams;
        cudaStream_t                    copy_streams[2];
        cudaStream_t                    kernel_streams[2];
        WindowInfo                      winfo;
        bool                            measuring;
        static bool                     timeout;
        
        enum { 
            COPY_STREAM1 = 0,
            COPY_STREAM2 = 1,
            KERNEL_STREAM = 2 
        };

        void register_config();
        bool open_config(arg_t file_mode = "rb");
        void close_config();
        void create_streams(cudaStream_t*& streams);

    public:

       ~Simulator();
        Simulator();
        Simulator(const string& path);

        // Getters.
        Tableau&        get_tableau() { return tableau; }
        Circuit&        get_circuit() { return circuit; }
        DeviceCircuit&  get_gpu_circuit() { return gpu_circuit; }
        bool            is_measuring() const { return measuring; }

        // Random circuit generation.
        Gatetypes get_rand_gate(const bool& multi_input = true, const bool& force_multi_input = false);
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
        void identity(Tableau& tab, const size_t& offset_per_partition, const size_t& num_qubits_per_partition, const cudaStream_t* streams, const InitialState& istate = Zero);

        // Advances the simulation by 1-time step.
        void step(const size_t& p, const depth_t& depth_level, const bool& reversed = false);
        
        // Do measurements in a single simulation step.
        void transpose(const bool& row_major, const cudaStream_t& stream);
        void reset_pivots(const size_t& num_pivots, const cudaStream_t& stream);
        void find_random_measures(const size_t& num_pivots, const cudaStream_t& stream);
        void compact_targets(const qubit_t& qubit, const cudaStream_t& stream);
        void inject_swap(const qubit_t& qubit, const sign_t& rbit, const cudaStream_t& stream);
        void inject_x(const qubit_t& qubit, const sign_t& rbit, const cudaStream_t& stream);
        void inject_cx(const uint32& active_targets, const cudaStream_t& stream);
        void tune_assuming_maximum_targets(const depth_t& depth_level);
        int64 measure_indeterminate(const depth_t& depth_level, const cudaStream_t& stream = 0);
        void measure(const size_t& p, const depth_t& depth_level, const bool& reversed = false);

        // Printers.
        void print_progress_header();
        void print_progress(const size_t& p, const depth_t& depth_level, const bool& passed = false);
        void print_tableau(const Tableau& tab, const depth_t& depth_level, const bool& reverse, const bool& prefix = false);
        void print_paulis(const Tableau& tab, const depth_t& depth_level, const bool& reversed);
        void print_gates(const DeviceCircuit& gates, const gate_ref_t& num_gates, const depth_t& depth_level);
        //void print_mesurements(const DeviceCircuit& gates, const gate_ref_t& num_gates, const depth_t& depth_level);

        // Timeout.
        static void handler_timeout(int) {
            fflush(stderr), fflush(stdout);
            LOG1("%s%s%s", CYELLOW, "TIME OUT", CNORMAL);
            timeout = true;
	    }

    };

}