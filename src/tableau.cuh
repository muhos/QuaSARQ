
#ifndef __CU_TABLEAU_H
#define __CU_TABLEAU_H

#include "table.cuh"
#include "signs.cuh"

namespace QuaSARQ {

    /*
    * DS encapsulating bit-encoded tables of Paulis and signs. 
    * 
    * See more info. about the bitencoding of XZ/Sign in 'Table' DS.
    */

    inline size_t get_num_partitions(const size_t& num_tableaus, const size_t& num_qubits, const size_t& extra_bytes, const size_t& max_free_memory) {
        const double free_accuracy = 0.995;
        const size_t corrected_free_memory = size_t(free_accuracy * max_free_memory);
        size_t num_qubits_padded = get_num_padded_bits(num_qubits);
        size_t num_words = get_num_words(num_qubits_padded * num_qubits_padded);
        size_t num_words_per_column = get_num_words(num_qubits);
        size_t max_padded_bits_two_tables = 2 * num_qubits_padded;
        assert(num_words_per_column * max_padded_bits_two_tables == 2 * num_words);
        size_t expected_capacity_required = num_tableaus * 2 * num_words * sizeof(word_std_t) + extra_bytes;
        size_t num_partitions = 1;
        while (expected_capacity_required >= corrected_free_memory && num_words_per_column > 1) {
            num_words_per_column = (num_words_per_column + 0.5) / 1.5;
            expected_capacity_required = num_tableaus * num_words_per_column * max_padded_bits_two_tables * sizeof(word_std_t) + extra_bytes;
            num_partitions++;
        }
        return num_partitions;
    }

#ifdef INTERLEAVE_XZ

    template <class ALLOCATOR>
    class Tableau {

        ALLOCATOR& allocator;

        Table* _ps;
        Signs* _ss;

        word_t* _ps_data;
        sign_t* _ss_data;

        size_t _num_words;
        size_t _num_qubits_padded;

        // Number of words encoding generators' bits.
        size_t _num_words_per_column;

        // Number of partitions spliting tableau generators' words.
        size_t _num_partitions;

    public:

        Tableau(ALLOCATOR& allocator) : 
            allocator(allocator) 
        ,   _ps(nullptr)
        ,   _ss(nullptr)
        ,   _ps_data(nullptr)
        ,   _ss_data(nullptr)
        ,   _num_qubits_padded(0)
        ,   _num_words(0)
        ,   _num_words_per_column(0)
        { }

        size_t alloc(const size_t& num_qubits, const size_t& max_window_bytes, const size_t& forced_num_partitions = 0) {
            if (!num_qubits)
                LOGERROR("cannot allocate tableau for 0 qubits.");
            if (_num_qubits_padded == get_num_padded_bits(num_qubits))
                return _num_partitions; 
            if (_num_qubits_padded) {
                // resize tableau.
                LOGERROR("Not yet implemented to resize a tableau.");
            }
            LOGN2(1, "Allocating tableau for %s%lld qubits%s.. ", CREPORTVAL, int64(num_qubits), CNORMAL);
            size_t cap_before = allocator.gpu_capacity();
            // Partition the tableau if needed.
            _num_qubits_padded = get_num_padded_bits(num_qubits);
            _num_words = get_num_words(_num_qubits_padded * _num_qubits_padded);
            _num_words_per_column = get_num_words(num_qubits);
            size_t max_padded_bits_two_tables = 2 * _num_qubits_padded;
            assert(_num_words_per_column * max_padded_bits_two_tables == 2 * _num_words);
            size_t expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + max_window_bytes;
            _num_partitions = 1;
            while ((forced_num_partitions && forced_num_partitions > _num_partitions) || (expected_capacity_required >= cap_before && _num_words_per_column > 1)) {
                _num_words_per_column = (_num_words_per_column + 0.5) / 1.5;
                expected_capacity_required = _num_words_per_column * max_padded_bits_two_tables * sizeof(word_std_t) + max_window_bytes;
                _num_partitions++;
            }
            if (forced_num_partitions && forced_num_partitions != _num_partitions) {
                LOGERRORN("insufficient number of partitions");
                throw GPU_memory_exception();
            }
            // Fix number of words for last partition.
			size_t partition_bits = _num_partitions * WORD_BITS;
			while ((partition_bits * _num_words_per_column) < _num_qubits_padded)
				_num_words_per_column++;
            // Update number of words.
			_num_words = _num_words_per_column * _num_qubits_padded;
			expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + max_window_bytes;       
            if (expected_capacity_required > cap_before) {
                LOGERRORN("insufficient memory");
                throw GPU_memory_exception();
            }
            assert(_num_partitions == 1 && _num_words_per_column == get_num_words(num_qubits)
                || _num_partitions > 1 && _num_partitions * _num_words_per_column <= _num_qubits_padded);
            // Create host pinned-memory objects to hold GPU pointers.
            Table *h_ps = new (allocator.template allocate_pinned<Table>(1)) Table();
            assert(h_ps != nullptr);
            Signs *h_ss = new (allocator.template allocate_pinned<Signs>(1)) Signs();
            assert(h_ss != nullptr);

            // Create CUDA memory for GPU pointers.
            _ps = allocator.template allocate<Table>(1);
            assert(_ps != nullptr);
            _ps_data = allocator.template allocate<word_t>(_num_words * 2);
            assert(_ps_data != nullptr);

            _ss = allocator.template allocate<Signs>(1);
            assert(_ss != nullptr);
            const size_t num_sign_words = _num_words_per_column;
            _ss_data = allocator.template allocate<sign_t>(num_sign_words);        
            assert(_ss_data != nullptr);

            // bind the allocated GPU pointers to the host object,
            // then transfer it to the GPU.
            h_ps->alloc(_num_words * 2, _num_words_per_column, _ps_data);
            h_ss->alloc(num_sign_words, _ss_data);

            CHECK(cudaMemcpyAsync(_ps, h_ps, sizeof(Table), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(_ss, h_ss, sizeof(Signs), cudaMemcpyHostToDevice));
            size_t cap_after = allocator.gpu_capacity();
            assert(cap_before > cap_after);
            size_t alloced = cap_before - cap_after;
            SYNCALL;
            LOGENDING(1, 3, "(reserved %zd MB, %zd partitions).", ratio(alloced, MB), _num_partitions);
            assert(_num_partitions);
            return _num_partitions;
        }

        void reset_signs() const {
            assert(_ss_data != nullptr);
            const size_t num_sign_words = _num_words_per_column;
            CHECK(cudaMemsetAsync(_ss_data, 0, num_sign_words * sizeof(sign_t)));
        }

        void reset() const {
            assert(_ps_data != nullptr);
            reset_signs();
            CHECK(cudaMemsetAsync(_ps_data, 0, 2 * _num_words * sizeof(word_t)));
        }

        INLINE_ALL size_t size() const { return 2 * _num_words + _num_words_per_column; }

        INLINE_ALL size_t num_words() const { return _num_words; }

        INLINE_ALL size_t num_qubits_padded() const { return _num_qubits_padded; }

        INLINE_ALL size_t num_words_per_column() const { return _num_words_per_column; }

        INLINE_ALL Signs* signs() const { assert(_ss != nullptr); return _ss; }

        INLINE_ALL Table* ptable() const { assert(_ps != nullptr); return _ps; }

        INLINE_ALL sign_t* signs_data() const { assert(_ss_data != nullptr); return _ss_data; }

        bool is_table_identity() const {
            Table tmp;
            CHECK(cudaMemcpy(&tmp, _ps, sizeof(Table), cudaMemcpyDeviceToHost));
            return tmp.is_identity();
        }

    };

#else

    template <class ALLOCATOR>
    class Tableau {

        ALLOCATOR& allocator;

        Table* _xs, * _zs;
        Signs* _ss;

        word_t* _xs_data;
        word_t* _zs_data;
        sign_t* _ss_data;

        size_t _num_words;
        size_t _num_qubits_padded;

        // Number of words encoding generators' bits
        size_t _num_words_per_column;

        // Number of partitions spliting tableau generators' words.
        size_t _num_partitions;

        // Tableau _extension to (2n) for measurements.
        size_t _extension;

    public:

        Tableau(ALLOCATOR& allocator) : 
            allocator(allocator) 
        ,   _xs(nullptr)
        ,   _zs(nullptr)
        ,   _ss(nullptr)
        ,   _xs_data(nullptr)
        ,   _zs_data(nullptr)
        ,   _ss_data(nullptr)
        ,   _num_words(0)
        ,   _num_qubits_padded(0)
        ,   _num_words_per_column(0)
        ,   _num_partitions(1)
        ,   _extension(1)
        { }

        size_t alloc(const size_t& num_qubits, const size_t& max_window_bytes, const bool& measuring = false, const size_t& forced_num_partitions = 0) {
            if (!num_qubits)
                LOGERROR("cannot allocate tableau for 0 qubits.");
            if (_num_qubits_padded == get_num_padded_bits(num_qubits))
                return _num_partitions; 
            if (_num_qubits_padded) {
                // resize tableau.
                LOGERROR("Not yet implemented to resize a tableau.");
            }
            LOGN2(1, "Allocating tableau for %s%lld qubits%s.. ", CREPORTVAL, int64(num_qubits), CNORMAL);
            size_t cap_before = allocator.gpu_capacity();
            // Partition the tableau if needed.
            _extension = measuring ? 2 : 1;
            _num_qubits_padded = get_num_padded_bits(num_qubits);
            _num_words_per_column = get_num_words(num_qubits * _extension);
            _num_words = _num_words_per_column * _num_qubits_padded;

            // TODO: rethink partitioning considering the extension.

            // size_t max_padded_bits_two_tables = 2 * _num_qubits_padded;
            // assert(_num_words_per_column * max_padded_bits_two_tables == 2 * _num_words);
            // size_t expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + max_window_bytes;
            // _num_partitions = 1;
            // while ((forced_num_partitions && forced_num_partitions > _num_partitions) || (expected_capacity_required >= cap_before && _num_words_per_column > 1)) {
            //     _num_words_per_column = (_num_words_per_column + 0.5) / 1.5;
            //     expected_capacity_required = _num_words_per_column * max_padded_bits_two_tables * sizeof(word_std_t) + max_window_bytes;
            //     _num_partitions++;
            // }
            // if (forced_num_partitions && forced_num_partitions != _num_partitions) {
            //     LOGERRORN("insufficient number of partitions");
            //     throw GPU_memory_exception();
            // }
            // // Fix number of words per column for last partition.
			// size_t partition_bits = _num_partitions * (WORD_BITS * _num_words_per_column);
			// while (partition_bits < _num_qubits_padded)
			// 	_num_words_per_column++;
            // // Update number of words.
			// _num_words = _num_words_per_column * _num_qubits_padded;
			// expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + max_window_bytes;       
            // if (expected_capacity_required > cap_before) {
            //     LOGERRORN("insufficient memory");
            //     throw GPU_memory_exception();
            // }
            // assert(_num_partitions == 1 && _num_words_per_column == get_num_words(num_qubits * _extension)
            //     || _num_partitions > 1 && _num_partitions * _num_words_per_column <= _num_qubits_padded);
            // Create host pinned-memory objects to hold GPU pointers.
            Table *h_xs = new (allocator.template allocate_pinned<Table>(1)) Table();
            assert(h_xs != nullptr);
            Table *h_zs = new (allocator.template allocate_pinned<Table>(1)) Table();
            assert(h_zs != nullptr);
            Signs *h_ss = new (allocator.template allocate_pinned<Signs>(1)) Signs();
            assert(h_ss != nullptr);

            // Create CUDA memory for GPU pointers.
            _xs = allocator.template allocate<Table>(1);
            assert(_xs != nullptr);
            _xs_data = allocator.template allocate<word_t>(_num_words);
            assert(_xs_data != nullptr);

            _zs = allocator.template allocate<Table>(1);
            assert(_zs != nullptr);
            _zs_data = allocator.template allocate<word_t>(_num_words);
            assert(_zs_data != nullptr);

            _ss = allocator.template allocate<Signs>(1);
            assert(_ss != nullptr);
            const size_t num_sign_words = _num_words_per_column;
            _ss_data = allocator.template allocate<sign_t>(num_sign_words);        
            assert(_ss_data != nullptr);

            // bind the allocated GPU pointers to the host object,
            // then transfer it to the GPU.
            h_xs->alloc(_num_words, _num_words_per_column, _xs_data);
            h_zs->alloc(_num_words, _num_words_per_column, _zs_data);
            h_ss->alloc(num_sign_words, _ss_data);

            CHECK(cudaMemcpyAsync(_xs, h_xs, sizeof(Table), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(_zs, h_zs, sizeof(Table), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(_ss, h_ss, sizeof(Signs), cudaMemcpyHostToDevice));
            size_t cap_after = allocator.gpu_capacity();
            assert(cap_before > cap_after);
            size_t alloced = cap_before - cap_after;
            SYNCALL;
            LOGENDING(1, 3, "(reserved %zd MB, %zd partitions).", ratio(alloced, MB), _num_partitions);
            assert(_num_partitions);
            return _num_partitions;
        }

        void reset_signs() const {
            assert(_ss_data != nullptr);
            const size_t num_sign_words = _num_words_per_column;
            CHECK(cudaMemsetAsync(_ss_data, 0, num_sign_words * sizeof(sign_t)));
        }

        void reset() const {
            assert(_xs_data != nullptr);
            assert(_zs_data != nullptr);
            reset_signs();
            CHECK(cudaMemsetAsync(_xs_data, 0, _num_words * sizeof(word_t)));
            CHECK(cudaMemsetAsync(_zs_data, 0, _num_words * sizeof(word_t)));
        }

        INLINE_ALL size_t size() const { return 2 * _num_words + _num_words_per_column; }

        INLINE_ALL size_t num_words() const { return _num_words; }

        INLINE_ALL size_t num_qubits_padded() const { return _num_qubits_padded; }

        INLINE_ALL size_t num_words_per_column() const { return _num_words_per_column; }

        INLINE_ALL Signs* signs() const { assert(_ss != nullptr); return _ss; }

        INLINE_ALL Table* xtable() const { assert(_xs != nullptr); return _xs; }

        INLINE_ALL Table* ztable() const { assert(_zs != nullptr); return _zs; }

        INLINE_ALL sign_t* signs_data() const { assert(_ss_data != nullptr); return _ss_data; }

        INLINE_ALL word_t* xtable_data() const { assert(_xs_data != nullptr); return _xs_data; }

        bool is_table_identity() const {
            Table tmp_zs, tmp_xs;
            CHECK(cudaMemcpy(&tmp_xs, _xs, sizeof(Table), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(&tmp_zs, _zs, sizeof(Table), cudaMemcpyDeviceToHost));
            return tmp_xs.is_identity() && tmp_zs.is_identity();
        }

    };

#endif

} 

#endif
