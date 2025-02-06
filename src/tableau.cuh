
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
        size_t num_words_major = get_num_words(num_qubits);
        size_t num_sign_words = num_words_major;
        size_t max_padded_bits_two_tables = 2 * num_qubits_padded;
        assert(num_words_major * max_padded_bits_two_tables == 2 * num_words);
        size_t expected_capacity_required = num_tableaus * 2 * num_words * sizeof(word_std_t) + num_sign_words * sizeof(sign_t) + extra_bytes;
        size_t num_partitions = 1;
        while (expected_capacity_required >= corrected_free_memory && num_words_major > 1) {
            num_words_major = (num_words_major + 0.5) / 1.5;
            num_sign_words = num_words_major;
            expected_capacity_required = num_tableaus * num_words_major * max_padded_bits_two_tables * sizeof(word_std_t) + num_sign_words * sizeof(sign_t) + extra_bytes;
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
        size_t _num_words_major;

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
        ,   _num_words_major(0)
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
            _num_words_major = get_num_words(num_qubits);
            size_t max_padded_bits_two_tables = 2 * _num_qubits_padded;
            assert(_num_words_major * max_padded_bits_two_tables == 2 * _num_words);
            size_t expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + max_window_bytes;
            _num_partitions = 1;
            while ((forced_num_partitions && forced_num_partitions > _num_partitions) || (expected_capacity_required >= cap_before && _num_words_major > 1)) {
                _num_words_major = (_num_words_major + 0.5) / 1.5;
                expected_capacity_required = _num_words_major * max_padded_bits_two_tables * sizeof(word_std_t) + max_window_bytes;
                _num_partitions++;
            }
            if (forced_num_partitions && forced_num_partitions != _num_partitions) {
                LOGERRORN("insufficient number of partitions");
                throw GPU_memory_exception();
            }
            // Fix number of words for last partition.
			size_t partition_bits = _num_partitions * WORD_BITS;
			while ((partition_bits * _num_words_major) < _num_qubits_padded)
				_num_words_major++;
            // Update number of words.
			_num_words = _num_words_major * _num_qubits_padded;
			expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + max_window_bytes;       
            if (expected_capacity_required > cap_before) {
                LOGERRORN("insufficient memory");
                throw GPU_memory_exception();
            }
            assert(_num_partitions == 1 && _num_words_major == get_num_words(num_qubits)
                || _num_partitions > 1 && _num_partitions * _num_words_major <= _num_qubits_padded);
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
            const size_t num_sign_words = _num_words_major;
            _ss_data = allocator.template allocate<sign_t>(num_sign_words);        
            assert(_ss_data != nullptr);

            // bind the allocated GPU pointers to the host object,
            // then transfer it to the GPU.
            h_ps->alloc(_num_words * 2, _num_words_major, _ps_data);
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
            const size_t num_sign_words = _num_words_major;
            CHECK(cudaMemsetAsync(_ss_data, 0, num_sign_words * sizeof(sign_t)));
        }

        void reset() const {
            assert(_ps_data != nullptr);
            reset_signs();
            CHECK(cudaMemsetAsync(_ps_data, 0, 2 * _num_words * sizeof(word_t)));
        }

        INLINE_ALL size_t size() const { return 2 * _num_words + _num_words_major; }

        INLINE_ALL size_t num_words_per_table() const { return _num_words; }

        INLINE_ALL size_t num_qubits_padded() const { return _num_qubits_padded; }

        INLINE_ALL size_t num_words_major() const { return _num_words_major; }

        INLINE_ALL Signs* signs() const { assert(_ss != nullptr); return _ss; }

        INLINE_ALL Table* ptable() const { assert(_ps != nullptr); return _ps; }

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

        Table* _h_xs, * _h_zs;
        Signs* _h_ss;

        word_t* _xs_data;
        word_t* _zs_data;
        sign_t* _ss_data;
        int* _unpacked_ss_data;

        size_t _num_qubits;
        size_t _num_qubits_padded;
        size_t _num_words;

        // Number of words encoding generators' bits in column-major.
        size_t _num_words_major;

        // Number of words encoding qubits' bits in row-major.
        size_t _num_words_minor;

        // Number of words encoding sign bits.
        size_t _num_sign_words;

        // Number of partitions spliting tableau generators' words.
        size_t _num_partitions;

        // Are signs unpacked?
        bool _unpacked_signs;

    public:

        Tableau(ALLOCATOR& allocator) : 
            allocator(allocator) 
        ,   _xs(nullptr)
        ,   _zs(nullptr)
        ,   _ss(nullptr)
        ,   _h_xs(nullptr)
        ,   _h_zs(nullptr)
        ,   _h_ss(nullptr)
        ,   _xs_data(nullptr)
        ,   _zs_data(nullptr)
        ,   _ss_data(nullptr)
        ,   _unpacked_ss_data(nullptr)
        ,   _num_qubits(0)
        ,   _num_qubits_padded(0)
        ,   _num_words(0)
        ,   _num_words_major(0)
        ,   _num_words_minor(0)
        ,   _num_sign_words(0)
        ,   _num_partitions(1)
        ,   _unpacked_signs(false)
        { }

        size_t alloc(const size_t& num_qubits, const size_t& max_window_bytes, const bool& prefix, const bool& measuring, const bool& unpack_signs = false, const size_t& forced_num_partitions = 0) {
            if (!num_qubits)
                LOGERROR("cannot allocate tableau for 0 qubits.");
            if (_num_qubits_padded == get_num_padded_bits(num_qubits))
                return _num_partitions; 
            if (_num_qubits_padded) {
                // reallocate tableau.
                LOGERROR("Not yet implemented to reallocate a tableau.");
            }
            LOGN2(1, "Allocating tableau for %s%lld qubits%s.. ", CREPORTVAL, int64(num_qubits), CNORMAL);
            size_t cap_before = allocator.gpu_capacity();
            _unpacked_signs = unpack_signs;
            size_t sign_word_size = _unpacked_signs ? sizeof(int) : sizeof(sign_t);
            // Partition the tableau if needed.
            _num_qubits = num_qubits;
            const size_t num_words_major_whole_tableau = get_num_words(_num_qubits);
            _num_qubits_padded = get_num_padded_bits(num_qubits);
            _num_words_major = num_words_major_whole_tableau;
            if (!prefix) _num_sign_words = _unpacked_signs ? _num_words_major * WORD_BITS : _num_words_major;
            _num_words = _num_words_major * _num_qubits_padded;
            const size_t max_padded_bits_two_tables = 2 * _num_qubits_padded;
            size_t expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
            _num_partitions = 1;
            assert(_num_words_major * max_padded_bits_two_tables == 2 * _num_words);
            while ((forced_num_partitions && forced_num_partitions > _num_partitions) || (expected_capacity_required >= cap_before && _num_words_major > 1)) {
                _num_words_major = (_num_words_major + 0.5) / 1.5;
                if (!prefix) _num_sign_words = _unpacked_signs ? _num_words_major * WORD_BITS : _num_words_major;
                expected_capacity_required = _num_words_major * max_padded_bits_two_tables * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
                _num_partitions++;
            }
            if (forced_num_partitions && forced_num_partitions != _num_partitions) {
                LOGERRORN("insufficient number of partitions.");
                throw GPU_memory_exception();
            }
            // Fix number of words per column for last partition.
			while ((_num_partitions * _num_words_major) < num_words_major_whole_tableau)
				_num_words_major++;
            // Update number of words.
            _num_words_minor = _num_words_major;
            if (measuring && !prefix) _num_words_major <<= 1;
            if (!prefix) _num_sign_words = _unpacked_signs ? _num_words_major * WORD_BITS : _num_words_major;
			_num_words = _num_words_major * _num_qubits_padded;
			expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;       
            if (expected_capacity_required > cap_before) {
                LOGERRORN("insufficient memory during tableau allocation.");
                throw GPU_memory_exception();
            }

            assert(_num_partitions == 1 && _num_words_major >= get_num_words((measuring && !prefix) ? 2 * _num_qubits : _num_qubits)
                || _num_partitions > 1 && _num_partitions * _num_words_major >= num_words_major_whole_tableau);
            
            // Create host pinned-memory objects to hold GPU pointers.
            _h_xs = new (allocator.template allocate_pinned<Table>(1)) Table();
            assert(_h_xs != nullptr);
            _h_zs = new (allocator.template allocate_pinned<Table>(1)) Table();
            assert(_h_zs != nullptr);
            if (!prefix) {
                _h_ss = new (allocator.template allocate_pinned<Signs>(1)) Signs();
                assert(_h_ss != nullptr);
            }

            // Create CUDA memory for GPU pointers.
            _xs = allocator.template allocate<Table>(1);
            assert(_xs != nullptr);
            _xs_data = allocator.template allocate<word_t>(_num_words);
            assert(_xs_data != nullptr);
            _zs = allocator.template allocate<Table>(1);
            assert(_zs != nullptr);
            _zs_data = allocator.template allocate<word_t>(_num_words);
            assert(_zs_data != nullptr);
            if (!prefix) {
                _ss = allocator.template allocate<Signs>(1);
                assert(_ss != nullptr);
            }

            // Bind the allocated GPU pointers to the host object,
            // then transfer it to the GPU.
            _h_xs->alloc(_xs_data, _num_qubits_padded, _num_words_major, _num_words_minor);
            _h_zs->alloc(_zs_data, _num_qubits_padded, _num_words_major, _num_words_minor);
            assert(_h_xs->size() == _num_words);
            assert(_h_zs->size() == _num_words);
            if (!prefix) {
                if (_unpacked_signs) {
                    _unpacked_ss_data = allocator.template allocate<int>(_num_sign_words);    
                    assert(_unpacked_ss_data != nullptr);
                    _h_ss->alloc(_unpacked_ss_data, _num_qubits_padded, _num_sign_words, true);
                }
                else {
                    _ss_data = allocator.template allocate<sign_t>(_num_sign_words);        
                    assert(_ss_data != nullptr);
                    _h_ss->alloc(_ss_data, _num_qubits_padded, _num_sign_words, false);
                }
                CHECK(cudaMemcpyAsync(_ss, _h_ss, sizeof(Signs), cudaMemcpyHostToDevice));
            }
            CHECK(cudaMemcpyAsync(_xs, _h_xs, sizeof(Table), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(_zs, _h_zs, sizeof(Table), cudaMemcpyHostToDevice));
            
            size_t cap_after = allocator.gpu_capacity();
            assert(cap_before > cap_after);
            size_t alloced = cap_before - cap_after;
            SYNCALL;
            LOGENDING(1, 3, "(reserved %zd MB, %zd partitions).", ratio(alloced, MB), _num_partitions);
            assert(_num_partitions);
            return _num_partitions;
        }

        // Doesn't reallocate memory.
        size_t resize(const size_t& num_qubits, const size_t& max_window_bytes, const bool& measuring = false, const bool& unpack_signs = false, const size_t& forced_num_partitions = 0) {
            if (!num_qubits)
                LOGERROR("cannot resize tableau for 0 qubits.");
            if (_num_qubits < num_qubits)
                LOGERROR("not enough memory for tableau resizing.");
            LOGN2(1, "Resizing tableau for %s%lld qubits%s.. ", CREPORTVAL, int64(num_qubits), CNORMAL);
            assert(_num_qubits >= num_qubits);
            // Reset the tableau.
            reset();
            _unpacked_signs = unpack_signs;
            size_t sign_word_size = _unpacked_signs ? sizeof(int) : sizeof(sign_t);
            size_t cap_before = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
            // Partition the tableau if needed.
            _num_qubits = num_qubits;
            const size_t num_words_major_whole_tableau = get_num_words(_num_qubits);
            _num_qubits_padded = get_num_padded_bits(num_qubits);
            _num_words_major = num_words_major_whole_tableau;
            _num_sign_words = _unpacked_signs ? _num_words_major * WORD_BITS : _num_words_major;
            _num_words = _num_words_major * _num_qubits_padded;
            const size_t max_padded_bits_two_tables = 2 * _num_qubits_padded;
            size_t expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
            _num_partitions = 1;
            assert(_num_words_major * max_padded_bits_two_tables == 2 * _num_words);
            while ((forced_num_partitions && forced_num_partitions > _num_partitions) || (expected_capacity_required >= cap_before && _num_words_major > 1)) {
                _num_words_major = (_num_words_major + 0.5) / 1.5;
                _num_sign_words = _unpacked_signs ? _num_words_major * WORD_BITS : _num_words_major;
                expected_capacity_required = _num_words_major * max_padded_bits_two_tables * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
                _num_partitions++;
            }
            if (forced_num_partitions && forced_num_partitions != _num_partitions) {
                LOGERRORN("insufficient number of partitions.");
                throw GPU_memory_exception();
            }
            // Fix number of words per column for last partition.
			while ((_num_partitions * _num_words_major) < num_words_major_whole_tableau)
				_num_words_major++;
                
            // Update number of words.
            _num_words_minor = _num_words_major;
            if (measuring) _num_words_major <<= 1;
			_num_words = _num_words_major * _num_qubits_padded;
			expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;       
            if (expected_capacity_required > cap_before) {
                LOGERRORN("insufficient memory during resizing.");
                throw GPU_memory_exception();
            }
            
            assert(_num_partitions == 1 && _num_words_major >= get_num_words(measuring ? 2 * _num_qubits : _num_qubits)
                || _num_partitions > 1 && _num_partitions * _num_words_major >= num_words_major_whole_tableau);
            
            // Bind the allocated GPU pointers to the host object,
            // then transfer it to the GPU.
            assert(_h_xs != nullptr && _h_zs != nullptr && _h_ss != nullptr);
            assert(_xs_data != nullptr);
            _h_xs->alloc(_xs_data, _num_qubits_padded, _num_words_major, _num_words_minor);
            assert(_zs_data != nullptr);
            _h_zs->alloc(_zs_data, _num_qubits_padded, _num_words_major, _num_words_minor);
            assert(_h_xs->size() == _num_words);
            assert(_h_zs->size() == _num_words);
            if (_unpacked_signs) {
                assert(_unpacked_ss_data != nullptr);
                _h_ss->alloc(_unpacked_ss_data, _num_qubits_padded, _num_sign_words, true);
            }
            else {      
                assert(_ss_data != nullptr);
                _h_ss->alloc(_ss_data, _num_qubits_padded, _num_sign_words, false);
            }
            CHECK(cudaMemcpyAsync(_xs, _h_xs, sizeof(Table), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(_zs, _h_zs, sizeof(Table), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(_ss, _h_ss, sizeof(Signs), cudaMemcpyHostToDevice));
            
            size_t cap_after = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
            assert(cap_before >= cap_after);
            size_t alloced = cap_before - cap_after;
            SYNCALL;
            LOGENDING(1, 3, "(reserved %zd MB, %zd partitions).", ratio(alloced, MB), _num_partitions);
            assert(_num_partitions);
            return _num_partitions;
        }

        void reset_signs() const {
            if (_unpacked_signs) {
                assert(_unpacked_ss_data != nullptr);
                CHECK(cudaMemsetAsync(_unpacked_ss_data, 0, _num_sign_words * sizeof(int)));
            }
            else {
                assert(_ss_data != nullptr);
                CHECK(cudaMemsetAsync(_ss_data, 0, _num_sign_words * sizeof(sign_t)));
            }
        }

        void reset() const {
            assert(_xs_data != nullptr);
            assert(_zs_data != nullptr);
            reset_signs();
            CHECK(cudaMemsetAsync(_xs_data, 0, _num_words * sizeof(word_t)));
            CHECK(cudaMemsetAsync(_zs_data, 0, _num_words * sizeof(word_t)));
        }

        INLINE_ALL size_t size() const { return 2 * _num_words + _num_sign_words; }

        INLINE_ALL size_t num_qubits() const { return _num_qubits; }

        INLINE_ALL size_t num_words_per_table() const { return _num_words; }

        INLINE_ALL size_t num_qubits_padded() const { return _num_qubits_padded; }

        INLINE_ALL size_t num_words_major() const { return _num_words_major; }

        INLINE_ALL size_t num_words_minor() const { return _num_words_minor; }

        INLINE_ALL Signs* signs() const { assert(_ss != nullptr); return _ss; }

        INLINE_ALL Table* xtable() const { assert(_xs != nullptr); return _xs; }

        INLINE_ALL Table* ztable() const { assert(_zs != nullptr); return _zs; }

        INLINE_ALL sign_t* sdata() { assert(_ss_data != nullptr); return _ss_data; }

        INLINE_ALL word_std_t* xdata(const size_t& offset = 0) { 
            assert(_xs_data != nullptr);
            assert(offset < _num_words);
            return reinterpret_cast<word_std_t*>(_xs_data) + offset;
        }

        INLINE_ALL word_std_t* zdata(const size_t& offset = 0) { 
            assert(_zs_data != nullptr); 
            assert(offset < _num_words);
            return reinterpret_cast<word_std_t*>(_zs_data) + offset;
        }

        bool is_table_identity() const {
            Table tmp_zs, tmp_xs;
            CHECK(cudaMemcpy(&tmp_xs, _xs, sizeof(Table), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(&tmp_zs, _zs, sizeof(Table), cudaMemcpyDeviceToHost));
            return tmp_xs.is_identity() && tmp_zs.is_identity();
        }

        void copy_to_host(Table* h_xs, Table* h_zs, Signs* h_ss = nullptr) {
            SYNCALL;
            if (h_xs != nullptr) {
                h_xs->alloc_host(_num_qubits_padded, _num_words_major, _num_words_minor);
                CHECK(cudaMemcpy(h_xs->data(), _xs_data, sizeof(word_t) * _num_words, cudaMemcpyDeviceToHost));
            }
            if (h_zs != nullptr) { 
                h_zs->alloc_host(_num_qubits_padded, _num_words_major, _num_words_minor);
                CHECK(cudaMemcpy(h_zs->data(), _zs_data, sizeof(word_t) * _num_words, cudaMemcpyDeviceToHost));
            }
            if (h_ss != nullptr) {
                h_ss->alloc_host(_num_qubits_padded, _num_sign_words, _unpacked_signs);
                if (_unpacked_signs)
                    CHECK(cudaMemcpy(h_ss->unpacked_data(), _unpacked_ss_data, sizeof(int) * _num_words, cudaMemcpyDeviceToHost));
                else
                    CHECK(cudaMemcpy(h_ss->data(), _ss_data, sizeof(sign_t) * _num_words, cudaMemcpyDeviceToHost));
            }
        }

        bool is_xstab_valid(const cudaStream_t& stream = 0) const { 
            Table tmp_xs;
            CHECK(cudaMemcpyAsync(&tmp_xs, _xs, sizeof(Table), cudaMemcpyDeviceToHost, stream));
            SYNC(stream);
            return tmp_xs.is_stab_valid();
        }

        bool is_xstab_valid(const qubit_t& q, const size_t& pivot, const cudaStream_t& stream = 0) const { 
            assert(_num_qubits);
            assert(_num_words_minor);
            const size_t stab_pivot = pivot + _num_qubits;
            const qubit_t q_w = WORD_OFFSET(q);
            const word_std_t q_mask = BITMASK_GLOBAL(q);
            assert(_xs_data != nullptr);
            assert((stab_pivot * _num_words_minor + q_w) < _num_words);
            const word_t* device_word_ptr = _xs_data + stab_pivot * _num_words_minor + q_w;
            word_t host_word;
            CHECK(cudaMemcpyAsync(&host_word, device_word_ptr, sizeof(word_t), cudaMemcpyDeviceToHost, stream));
            SYNC(stream);
            //printf("pivot: %d, word = " B2B_STR "\n", stab_pivot, RB2B(word_std_t(host_word)));
            return (word_std_t(host_word) & q_mask);
        }

    };

#endif

} 

#endif
