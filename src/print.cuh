#ifndef __CU_PRINT_H
#define __CU_PRINT_H

#include "table.cuh"
#include "signs.cuh"
#include "vector.cuh"
#include "locker.cuh"
#include "circuit.cuh"
#include "grid.cuh"

namespace QuaSARQ {

    NOINLINE_DEVICE void REPCH_GPU(const char* ch, const size_t& size, const size_t& off = 0);

    NOINLINE_ALL void print_table(const Table& t);

    NOINLINE_ALL void print_table(const Table& t, const size_t& num_qubits, const size_t& num_words_major, const size_t& num_words_minor);

    NOINLINE_ALL void print_table_signs(const Signs& ss, const size_t& offset = 0);

    NOINLINE_ALL void print_tables(const Table& xs, const Table& zs, const Signs& ss, const size_t& num_qubits, const int64& level, const bool& measuring);

    NOINLINE_ALL void print_tables(const Table& ps, const Signs& ss, const size_t& num_qubits, const int64& level, const bool& measuring);

    NOINLINE_ALL void print_state(const Table& xs, const Table& zs, const Signs& ss, const size_t& start, const size_t& end, const size_t& num_qubits, const size_t& num_words_major);

    NOINLINE_DEVICE void print_column(DeviceLocker& dlocker, const Table& xs, const Table& zs, const Signs& ss, const size_t& q, const size_t& num_qubits, const size_t& num_words_major);

    NOINLINE_DEVICE void print_row(DeviceLocker& dlocker, const Gate& m, const Table& inv_xs, const Table& inv_zs, const Signs& inv_ss, const size_t& row, const size_t& num_words_minor);

    NOINLINE_DEVICE void print_shared_aux(DeviceLocker& dlocker, const Gate& m, byte_t* smem, const size_t& copied_row, const size_t& multiplied_row = UINT64_MAX);

    // Print the tableau in binary format (generators are columns).
    __global__ void print_tableau_k(const Table* xs, const Table* zs, const Signs* ss, const size_t num_qubits, const depth_t level, const bool measuring);
    __global__ void print_tableau_k(const Table* ps, const Signs* ss, const size_t num_qubits, const depth_t level, const bool measuring);

    // Print the tableau's Pauli strings.
    __global__ void print_paulis_k(const Table* xs, const Table* zs, const Signs* ss, const size_t num_words_major, const size_t num_qubits, const bool extended);
    __global__ void print_paulis_k(const Table* ps, const Signs* ss, const size_t num_words_major, const size_t num_qubits, const bool extended);

    // Print gates.
    __global__ void print_gates_k(const gate_ref_t* refs, const bucket_t* gates, const Pivot* pivots, const gate_ref_t num_gates);

    // Print measurements.
    __global__ void print_measurements_k(const gate_ref_t* refs, const bucket_t* measurements, const Pivot* pivots, const gate_ref_t num_gates);

}

#endif