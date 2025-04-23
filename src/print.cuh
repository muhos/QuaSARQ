#pragma once

#include "datatypes.cuh"
#include "circuit.cuh"
#include "vector.cuh"
#include "locker.cuh"
#include "grid.cuh"

namespace QuaSARQ {

    NOINLINE_DEVICE void REPCH_GPU(const char* ch, const size_t& size, const size_t& off = 0);

    NOINLINE_ALL void print_table(const Table& t, const size_t& total_targets = 0);

    NOINLINE_ALL void print_table_signs(const Signs& ss, const size_t& start, const size_t& end);

    NOINLINE_ALL void print_tables(const Table& xs, const Table& zs, const Signs* ss, const int64& level);

    NOINLINE_ALL void print_state(const Table& xs, const Table& zs, const Signs& ss, const size_t& start, const size_t& end, const size_t& num_qubits, const size_t& num_words_major);

    NOINLINE_DEVICE void print_column(DeviceLocker& dlocker, const Table& xs, const Table& zs, const Signs& ss, const size_t& q, const size_t& num_qubits, const size_t& num_words_major);

    NOINLINE_DEVICE void print_row(DeviceLocker& dlocker, const Gate& m, const Table& inv_xs, const Table& inv_zs, const Signs& inv_ss, const size_t& row, const size_t& num_words_minor);

    // Print the tableau in binary format (generators are columns).
    __global__ void print_tableau_k(const_table_t xs, const_table_t zs, const_signs_t ss, const depth_t level);

    // Print the tableau's Pauli strings.
    __global__ void print_paulis_k(const_table_t xs, const_table_t zs, const_signs_t ss, const size_t num_words_major, const size_t num_qubits, const bool extended);
    __global__ void print_paulis_k(const_table_t ps, const_signs_t ss, const size_t num_words_major, const size_t num_qubits, const bool extended);

    // Print gates.
    __global__ void print_gates_k(const_refs_t refs, const_buckets_t gates, const_pivots_t pivots, const gate_ref_t num_gates);

    // Print measurements.
    __global__ void print_measurements_k(const_refs_t refs, const_buckets_t measurements, const_pivots_t pivots, const gate_ref_t num_gates);

}