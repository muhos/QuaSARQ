#pragma once

#include <string>

#include "definitions.cuh"
#include "memory.cuh"
#include "malloc.hpp"
#include "word.cuh"
#include "access.cuh"

namespace QuaSARQ {

    #define BIT_OFFSET(IDX)  ((IDX) & WORD_MASK)
    #define WORD_OFFSET(IDX) ((IDX) >> WORD_POWER)

    #define Z_TABLE(TABLEAU)  TABLEAU.ztable()
    #define X_TABLE(TABLEAU)  TABLEAU.xtable()
    #define XZ_TABLE(TABLEAU) TABLEAU.xtable(), TABLEAU.ztable()

    /*
    * A flat 2D matrix of bit-packed words of size 'word_t'.
    * Benchmarking showed that 64-bit words are the best
    * for GPU memory accesses and multiprocessors occupency.
    * For flexibility, 'word_t' is created to define any other
    * data type e.g. 16- or 32-bit integer.
    * 
    * The 'generators' matrix is accessed in a column-major
    * fashion in order to update the tableau in a consecutive
    * manner for all words per column. This makes the update
    * cache friendly.
    * 
    * Therefore to update generators for qubit q, we do the
    * following algorithm:
    * 
    * forall w: 0 to n do (in parallel): 
    *    OPERATION(generators[q * num_words_major + w])
    * 
    * To access words per column (for all paulis), for instance
    * to calculate the signs:
    * 
    * forall w: 0 to n do (in parallel): 
    *    OPERATION(generators[q + w * num_words_major])
    * 
    */
    class Table {

        word_t* _data;
        size_t _num_qubits_padded;
        size_t _num_words;
        size_t _num_words_minor;
        size_t _num_words_major;
        bool _is_identity;
        bool _is_rowmajor;
        Context _context;

        INLINE_ALL
        size_t get_diagonal_index(const qubit_t& q, const qubit_t& column_offset = 0) const {
            const size_t idx = (q + column_offset) * _num_words_major +
                                WORD_OFFSET(q);
            assert(idx < _num_words);
            return idx;
        }

        INLINE_ALL
        size_t get_diagonal_index_off(const qubit_t& q, const qubit_t& column_offset = 0) const {
            const size_t idx = (q + column_offset) * _num_words_major + 
                                WORD_OFFSET(q + _num_qubits_padded);
            assert(idx < _num_words);
            return idx;
        }

    public:

        Table() :
            _data(nullptr)
            , _num_qubits_padded(0)
            , _num_words(0)
            , _num_words_minor(0)
            , _num_words_major(0)
            , _is_identity(true)
            , _is_rowmajor(false)
            , _context(UNKNOWN)
        { }

        ~Table() {
            destroy();
        }

        void destroy() {
            if (_context == CPU) {
                if (_data != nullptr) {
                    std::free(_data);
                }
            }
            _data = nullptr;
            _is_identity = true;
            _context = UNKNOWN;
        }

        void alloc(word_t* data_ptr, const size_t& num_qubits_padded, const size_t& num_words_major, const size_t& num_words_minor) {
            if (_context == CPU) {
                LOGGPUERROR("cannot assign GPU pointer to a pre-allocated CPU pointer.");
                return;
            }
            assert(num_qubits_padded);
            assert(num_words_major);
            assert(data_ptr != nullptr);
            _num_qubits_padded = num_qubits_padded;
            _num_words_major = num_words_major;
            _num_words_minor = num_words_minor;
            _num_words = _num_words_major * _num_qubits_padded;
            _data = data_ptr;
            assert(_is_identity == true);
            _context = GPU;
        }

        void alloc_host(const size_t& num_qubits_padded, const size_t& num_words_major, const size_t& num_words_minor) {
            if (_context == GPU) {
                LOGERRORN("cannot allocate CPU pointer to a pre-allocated GPU pointer.");
                return;
            }
            if (_data != nullptr) {
                assert(_num_words == num_words_major * num_qubits_padded);
                return;
            }
            assert(num_qubits_padded);
            assert(num_words_major);
            _num_qubits_padded = num_qubits_padded;
            _num_words_major = num_words_major;
            _num_words_minor = num_words_minor;
            _num_words = _num_words_major * _num_qubits_padded;
            _data = calloc<word_t>(_num_words);
            _context = CPU;
        }

        INLINE_ALL size_t num_qubits_padded() const { return _num_qubits_padded; }

        INLINE_ALL size_t size() const { return _num_words; }

        INLINE_ALL size_t num_words_major() const { return _num_words_major; }

        INLINE_ALL size_t num_words_minor() const { return _num_words_minor; }

        INLINE_ALL word_t* data() { return _data; }

        INLINE_ALL const word_t* data() const { return _data; }

        INLINE_ALL void set_destab_to_identity(const qubit_t& q, const qubit_t& column_offset = 0) {
            _data[get_diagonal_index(q, column_offset)].identity(q);
        }

        INLINE_ALL void set_stab_to_identity(const qubit_t& q, const qubit_t& column_offset = 0) {
            _data[get_diagonal_index_off(q, column_offset)].identity(q + _num_qubits_padded);
        }

        INLINE_ALL void set_word_to_identity(const qubit_t& q, const qubit_t& column_offset = 0) {
            _data[get_diagonal_index(q, column_offset)].identity(q);
        }

        INLINE_ALL bool check_stab_is_identity(const qubit_t& q, const qubit_t& column_offset) const {
            return _data[get_diagonal_index_off(q, column_offset)].is_identity(q + _num_qubits_padded);
        }

        INLINE_ALL bool check_destab_is_identity(const qubit_t& q, const qubit_t& column_offset) const {
            return _data[get_diagonal_index(q, column_offset)].is_identity(q);
        }

        INLINE_ALL bool check_word_is_identity(const qubit_t& q, const qubit_t& column_offset) const {
            return _data[get_diagonal_index(q, column_offset)].is_identity(q);
        }

        // Same as data() but returns 'word_std_t' value at index 'idx'.
        INLINE_ALL word_std_t* words(const size_t& idx = 0) {          
            assert(idx < _num_words);
            return reinterpret_cast<word_std_t*>(_data) + idx;
        }

        // Same as data() but returns 'word_std_t' value at index 'idx'.
        INLINE_ALL const word_std_t* words(const size_t& idx = 0) const { 
            assert(idx < _num_words);
            return reinterpret_cast<word_std_t*>(_data) + idx;
        }

        // Return table 'word_t' type.
        INLINE_ALL const word_t operator[] (const size_t& idx) const {
            assert(idx < _num_words);
            return _data[idx];
        }

        // Return table 'word_t' type.
        INLINE_ALL word_t& operator[] (const size_t& idx) {
            assert(idx < _num_words);
            return _data[idx];
        }

        INLINE_ALL void flag_not_indentity() {
            if (_is_identity) {
                _is_identity = false;
            }
        }

        INLINE_ALL void flag_rowmajor() {
            if (!_is_rowmajor) {
                _is_rowmajor = true;
            }
        }

        INLINE_ALL void flag_colmajor() {
            if (_is_rowmajor) {
                _is_rowmajor = false;
            }
        }

        INLINE_ALL void flag_orientation(const bool& row_major) {
            _is_rowmajor = row_major;
        }

        INLINE_ALL bool is_identity() const { return _is_identity; }

        INLINE_ALL bool is_rowmajor() const { return _is_rowmajor; }

        INLINE_ALL bool is_colmajor() const { return !_is_rowmajor; }

    };    

}