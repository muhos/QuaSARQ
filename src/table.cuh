

#ifndef __CU_TABLE_H
#define __CU_TABLE_H

#include <string>

#include "definitions.cuh"
#include "memory.cuh"
#include "malloc.hpp"
#include "word.cuh"
#include "macros.cuh"

namespace QuaSARQ {

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
    *    OPERATION(generators[q + w * num_words_minor])
    * 
    */

    #define BIT_OFFSET(IDX)  ((IDX) & WORD_MASK)
    #define WORD_OFFSET(IDX) ((IDX) >> WORD_POWER)

    // Used in interleaving mode.
#ifdef INTERLEAVE_XZ
    #ifdef INTERLEAVE_WORDS
        #define X_WORD_OFFSET(W) ((W) << 1)
        #define Z_WORD_OFFSET(W) (X_WORD_OFFSET(W) | 1)
        #define X_OFFSET(IDX) X_WORD_OFFSET(IDX)
        #define Z_OFFSET(IDX) X_WORD_OFFSET(IDX)
    #else
        #define X_WORD_OFFSET(W) (W)
        #define Z_WORD_OFFSET(W) (W)
        constexpr size_t INTERLEAVE_OFFSET =  2 * INTERLEAVE_COLS;
        INLINE_ALL size_t X_OFFSET(const size_t& idx) {
            return (((idx) / INTERLEAVE_COLS) * INTERLEAVE_OFFSET) + idx % INTERLEAVE_COLS;
        }
        INLINE_ALL size_t Z_OFFSET(const size_t& idx) { 
            return X_OFFSET(idx) + INTERLEAVE_COLS;
        } 
    #endif
    #define Z_TABLE(TABLEAU)  TABLEAU.ptable()
    #define X_TABLE(TABLEAU)  TABLEAU.ptable()
    #define XZ_TABLE(TABLEAU) TABLEAU.ptable()
#else
    #define X_WORD_OFFSET(W) (W)
    #define Z_WORD_OFFSET(W) (W)
    #define X_OFFSET(IDX)    (IDX)
    #define Z_OFFSET(IDX)    (IDX)
    #define Z_TABLE(TABLEAU)  TABLEAU.ztable()
    #define X_TABLE(TABLEAU)  TABLEAU.xtable()
    #define XZ_TABLE(TABLEAU) TABLEAU.xtable(), TABLEAU.ztable()
#endif

    class Table {

        word_t* _data;
        size_t _num_words;
        size_t _num_words_major;
        byte_t  _is_identity;
        Context _context;

    public:

        Table() :
            _data(nullptr)
            , _num_words(0)
            , _num_words_major(0)
            , _is_identity(1)
            , _context(UNKNOWN)
        { }

        ~Table() {
            destroy();
        }

        void destroy() {
            if (_context == CPU) {
                if (_data != nullptr)
                    std::free(_data);
            }
            _num_words = 0;
            _num_words_major = 0;
            _data = nullptr;
        }

        void alloc(const size_t& num_words, const size_t& num_words_major, word_t* data_ptr) {
            if (_context == CPU) {
                LOGGPUERROR("cannot assign GPU pointer to a pre-allocated CPU pointer.");
                return;
            }
            assert(num_words);
            assert(num_words_major);
            assert(data_ptr != nullptr);
            _num_words = num_words;
            _num_words_major = num_words_major;
            _data = data_ptr;
            assert(_is_identity == 1);
            _context = GPU;
        }

        void alloc_host(const size_t& num_words, const size_t& num_words_major) {
            if (_context == GPU) {
                LOGERRORN("cannot allocate CPU pointer to a pre-allocated GPU pointer.");
                return;
            }
            assert(num_words);
            assert(num_words_major);
            _num_words = num_words;
            _num_words_major = num_words_major;
            _data = calloc<word_t>(_num_words);
            _context = CPU;
        }

        INLINE_ALL size_t size() const { return _num_words; }

        INLINE_ALL size_t num_words_major() const { return _num_words_major; }

        INLINE_ALL word_t* data() { return _data; }

        INLINE_ALL const word_t* data() const { return _data; }

        INLINE_ALL void set_x_word_to_identity(const qubit_t& q, const qubit_t& offset = 0) {
            const size_t idx = (X_OFFSET(q) + offset) * _num_words_major + X_WORD_OFFSET(WORD_OFFSET(q));
            assert(idx < _num_words);
            _data[idx].identity(q);
        }

        INLINE_ALL void set_z_word_to_identity(const qubit_t& q, const qubit_t& offset = 0) {
            const size_t idx = (Z_OFFSET(q) + offset) * _num_words_major + Z_WORD_OFFSET(WORD_OFFSET(q));
            assert(idx < _num_words);
            _data[idx].identity(q);
        }

        INLINE_ALL void set_word_to_identity(const qubit_t& q, const qubit_t& offset = 0) {
            const size_t idx = (q + offset) * _num_words_major + WORD_OFFSET(q);
            assert(idx < _num_words);
            _data[idx].identity(q);
        }

        INLINE_ALL word_t* words(const qubit_t& q) {          
            const size_t idx = q * _num_words_major;
            assert(idx < _num_words);
            return _data + idx;
        }

        INLINE_ALL const word_t* words(const qubit_t& q) const { 
            const size_t idx = q * _num_words_major;
            assert(idx < _num_words);
            return _data + idx;
        }

        // 'idx' is a matrix index.
        // Return table word.
        INLINE_ALL const word_t operator[] (const size_t& idx) const {
            assert(idx < _num_words);
            return _data[idx];
        }

        // 'idx' is a matrix index.
       // Return table word.
        INLINE_ALL word_t& operator[] (const size_t& idx) {
            assert(idx < _num_words);
            return _data[idx];
        }

        INLINE_ALL bool check_z_word_is_identity(const qubit_t& q, const qubit_t& offset) const {
            const size_t idx = (Z_OFFSET(q) + offset) * _num_words_major + WORD_OFFSET(q);
            assert(idx < _num_words);
            return _data[idx].is_identity(q);
        }

        INLINE_ALL bool check_x_word_is_identity(const qubit_t& q, const qubit_t& offset) const {
            const size_t idx = (X_OFFSET(q) + offset) * _num_words_major + WORD_OFFSET(q);
            assert(idx < _num_words);
            return _data[idx].is_identity(q);
        }

        INLINE_ALL bool check_word_is_identity(const qubit_t& q, const qubit_t& offset) const {
            const size_t idx = (q + offset) * _num_words_major + WORD_OFFSET(q);
            assert(idx < _num_words);
            return _data[idx].is_identity(q);
        }

        INLINE_ALL void flag_not_indentity() {
            if (_is_identity == 1)
                _is_identity = 0;
        }

        INLINE_ALL bool is_identity() const { return _is_identity; }

    };    

}

#endif
