#ifndef __SIMULATOR_H
#define __SIMULATOR_H

#include "vector.cuh"
#include "locker.cuh"
#include "tableau.cuh"
#include "circuit.cuh"
#include "prefix.cuh"
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
        Vec<qubit_t, size_t>            measurements;
        Vec<qubit_t, size_t>            shuffled;
        Vec<byte_t, size_t>             locked;
        DeviceAllocator                 gpu_allocator;
        DeviceCircuit                   gpu_circuit;
        Locker                          locker;
        Tableau                         tableau;
        Tableau                         inv_tableau;
        Commutation*                    commutations;
        Commutations                    commuting; 
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
        void find_pivots(const size_t& num_pivots, const cudaStream_t& stream);
        void compact_targets(const qubit_t& qubit, const cudaStream_t& stream);
        void mark_commutations(const qubit_t& qubit, const cudaStream_t& stream);
        void inject_swap(const qubit_t& qubit, const cudaStream_t& stream);
        void inject_cx(const uint32& active_targets, const cudaStream_t& stream);
        void tune_assuming_maximum_targets(const depth_t& depth_level);
        int64 measure_indeterminate(const depth_t& depth_level, const cudaStream_t& stream = 0);
        void measure(const size_t& p, const depth_t& depth_level, const bool& reversed = false);

        // Printers.
        void print_tableau(const Tableau& tab, const depth_t& depth_level, const bool& reverse, const bool& prefix = false);
        void print_paulis(const Tableau& tab, const depth_t& depth_level, const bool& reversed);
        void print_gates(const DeviceCircuit& gates, const gate_ref_t& num_gates, const depth_t& depth_level);
        void print_measurements(const DeviceCircuit& gates, const gate_ref_t& num_gates, const depth_t& depth_level);

        // Progress report.
        inline void print_progress_header() {
            LOG2(1, "   %-10s    %-10s    %-10s    %15s          %-15s", 
                    "Partition", "Step", "Gates", "Measurements", "Time (s)");
            LOG2(1, "   %-10s    %-10s    %-10s    %-10s  %-10s    %-10s", 
                    "", "", "", "definite", "random", "");
            LOGRULER('-', RULERLEN);
        }
        inline void print_progress(const size_t& p, const depth_t& depth_level) {
            if (options.progress_en) {
                progress_timer.stop();
                const bool is_measuring = circuit.is_measuring(depth_level);
                size_t random_measures = stats.circuit.measure_stats.random_per_window;
                stats.circuit.measure_stats.random_per_window = 0;
                size_t prev_num_gates = circuit[depth_level].size();
                size_t definite_measures = is_measuring ? prev_num_gates - random_measures : 0;
                if (is_measuring) SETCOLOR(CLBLUE, stdout);
                else SETCOLOR(CORANGE1, stdout);
                LOG2(1, "%c  %-10lld    %-10lld    %-10lld    %-10lld  %-10lld   %-7.3f", 
                        is_measuring ? 'm' : 'u',
                        p + 1, depth_level + 1, prev_num_gates, definite_measures, random_measures, progress_timer.time() / 1000.0);
                SETCOLOR(CNORMAL, stdout);
            }
        }

    };

}

#endif