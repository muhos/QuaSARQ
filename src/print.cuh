#pragma once

#include "datatypes.cuh"
#include "circuit.cuh"
#include "vector.cuh"
#include "locker.cuh"
#include "grid.cuh"

namespace QuaSARQ {

    NOINLINE_ALL void REPCH_GPU(const char* ch, const size_t& size, const size_t& off = 0);

    NOINLINE_ALL void print_table(const Table& t, const size_t& total_targets = 0);

    NOINLINE_ALL void print_table_signs(const Signs& ss, const size_t& start, const size_t& end);

    NOINLINE_ALL void print_tables(const Table& xs, const Table& zs, const Signs* ss, const int64& level);

    NOINLINE_ALL void print_state(const Table& xs, const Table& zs, const Signs& ss, const size_t& start, const size_t& end, const size_t& num_qubits, const size_t& num_words_major);

    NOINLINE_DEVICE void print_column(DeviceLocker& dlocker, const Table& xs, const Table& zs, const Signs& ss, const size_t& q, const size_t& num_qubits, const size_t& num_words_major);

    NOINLINE_DEVICE void print_row(DeviceLocker& dlocker, const Gate& m, const Table& inv_xs, const Table& inv_zs, const Signs& inv_ss, const size_t& row, const size_t& num_words_minor);

}