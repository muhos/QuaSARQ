#ifndef __CU_SIGNS_H
#define __CU_SIGNS_H

#include "word.cuh"
#include "macros.cuh"
#include "malloc.hpp"
#include "logging.hpp"

namespace QuaSARQ {

    typedef word_std_t sign_t;

    class Signs {

        sign_t* _data;
        size_t _num_words;
        Context _context;

    public:

        Signs() :
            _data(nullptr)
            , _num_words(0)
            , _context(UNKNOWN)
        { }

        ~Signs() {
            destroy();
        }

        void destroy() {
            if (_context == CPU) {
                if (_data != nullptr)
                    std::free(_data);
            }
            _num_words = 0;
            _data = nullptr;
        }

        // Doesn't allocate memory by itself. Memory should
        // be allocated and assigned to 'data_ptr'.
        void alloc(const size_t& num_words, sign_t* data_ptr) {
            if (_context == CPU) {
                LOGGPUERROR("cannot assign GPU pointer to a pre-allocated CPU pointer.");
                return;
            }
            assert(data_ptr != nullptr);
            _data = data_ptr;
            _num_words = num_words;
            _context = GPU;
        }

        void alloc_host(const size_t& num_words) {
            if (_context == GPU) {
                LOGERRORN("cannot allocate CPU pointer to a pre-allocated GPU pointer.");
                return;
            }
            _num_words = num_words;
            _data = calloc<sign_t>(_num_words);
            _context = CPU;
        }

        INLINE_ALL size_t size() const { return _num_words; }

        // Return a pointer to q's sign.
        INLINE_ALL sign_t* data(const qubit_t& q = 0) { 
            assert(q < _num_words);
            return _data + q;
        }

        // Return a pointer to q's sign.
        INLINE_ALL const sign_t* data(const qubit_t& q = 0) const {
            assert(q < _num_words);
            return _data + q;
        }

        // Return sign word.
        INLINE_ALL sign_t& operator[] (const size_t& q) {
            assert(q < _num_words);
            return _data[q];
        }

        // Return sign word.
        INLINE_ALL sign_t operator[] (const size_t& q) const {
            assert(q < _num_words);
            return _data[q];
        }

    };

}


#endif