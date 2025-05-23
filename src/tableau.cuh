
#pragma once

#include "table.cuh"
#include "signs.cuh"

namespace QuaSARQ {

   

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

    /*
    * DS encapsulating bit-encoded tables of Paulis and signs. 
    * 
    * See more info. about the bitencoding of XZ/Sign in 'Table' DS.
    */
    class Tableau {

        DeviceAllocator& allocator;

        Table* _xs, * _zs;
        Signs* _ss;

        Table* _h_xs, * _h_zs;
        Signs* _h_ss;

        word_t* _xs_data;
        word_t* _zs_data;
        sign_t* _ss_data;

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

        void swap_tables(Tableau &lhs, Tableau &rhs) {
            using std::swap;
            swap(lhs._xs,      rhs._xs);
            swap(lhs._zs,      rhs._zs);
            swap(lhs._h_xs,    rhs._h_xs);
            swap(lhs._h_zs,    rhs._h_zs);
            swap(lhs._xs_data, rhs._xs_data);
            swap(lhs._zs_data, rhs._zs_data);
        }

    public:

        Tableau(DeviceAllocator& allocator) : 
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
        ,   _num_qubits(0)
        ,   _num_qubits_padded(0)
        ,   _num_words(0)
        ,   _num_words_major(0)
        ,   _num_words_minor(0)
        ,   _num_sign_words(0)
        ,   _num_partitions(1)
        { }

        void swap_tableaus(Tableau &other) {
            swap_tables(*this, other);
        }

        size_t alloc(const size_t& num_qubits, const size_t& max_window_bytes, const bool& prefix, const bool& measuring, const bool& alloc_signs, const size_t& forced_num_partitions = 0) {
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
            size_t sign_word_size = sizeof(sign_t);
            // Partition the tableau if needed.
            _num_qubits = num_qubits;
            const size_t num_words_major_whole_tableau = get_num_words(_num_qubits);
            _num_qubits_padded = get_num_padded_bits(num_qubits);
            _num_words_major = num_words_major_whole_tableau;
            if (!prefix && alloc_signs) _num_sign_words = _num_words_major;
            _num_words = _num_words_major * _num_qubits_padded;
            const size_t max_padded_bits_two_tables = 2 * _num_qubits_padded;
            size_t expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
            _num_partitions = 1;
            assert(_num_words_major * max_padded_bits_two_tables == 2 * _num_words);
            while ((forced_num_partitions && forced_num_partitions > _num_partitions) || (expected_capacity_required >= cap_before && _num_words_major > 1)) {
                _num_words_major = (_num_words_major + 0.5) / 1.5;
                if (!prefix && alloc_signs) _num_sign_words = _num_words_major;
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
            if (!prefix && alloc_signs) _num_sign_words = _num_words_major;
			_num_words = _num_words_major * _num_qubits_padded;
			expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;       
            if (expected_capacity_required > cap_before) {
                LOGERRORN("insufficient memory during tableau allocation.");
                throw GPU_memory_exception();
            }

            assert(_num_partitions == 1 && _num_words_major >= get_num_words((measuring && !prefix) ? 2 * _num_qubits : _num_qubits)
                || _num_partitions > 1 && _num_partitions * _num_words_major >= num_words_major_whole_tableau);
            
            // Create host pinned-memory objects to hold GPU pointers.
            _h_xs = new (allocator.allocate_pinned<Table>(1)) Table();
            assert(_h_xs != nullptr);
            _h_zs = new (allocator.allocate_pinned<Table>(1)) Table();
            assert(_h_zs != nullptr);
            if (!prefix && alloc_signs) {
                _h_ss = new (allocator.allocate_pinned<Signs>(1)) Signs();
                assert(_h_ss != nullptr);
            }

            // Create CUDA memory for GPU pointers.
            _xs = allocator.allocate<Table>(1);
            assert(_xs != nullptr);
            _xs_data = allocator.allocate<word_t>(_num_words);
            assert(_xs_data != nullptr);
            _zs = allocator.allocate<Table>(1);
            assert(_zs != nullptr);
            _zs_data = allocator.allocate<word_t>(_num_words);
            assert(_zs_data != nullptr);
            if (!prefix && alloc_signs) {
                _ss = allocator.allocate<Signs>(1);
                assert(_ss != nullptr);
            }

            // Bind the allocated GPU pointers to the host object,
            // then transfer it to the GPU.
            _h_xs->alloc(_xs_data, _num_qubits_padded, _num_words_major, _num_words_minor);
            _h_zs->alloc(_zs_data, _num_qubits_padded, _num_words_major, _num_words_minor);
            assert(_h_xs->size() == _num_words);
            assert(_h_zs->size() == _num_words);
            if (!prefix && alloc_signs) {
                _ss_data = allocator.allocate<sign_t>(_num_sign_words);        
                assert(_ss_data != nullptr);
                _h_ss->alloc(_ss_data, _num_qubits_padded, _num_sign_words, false);
                CHECK(cudaMemcpyAsync(_ss, _h_ss, sizeof(Signs), cudaMemcpyHostToDevice));
            }
            CHECK(cudaMemcpyAsync(_xs, _h_xs, sizeof(Table), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(_zs, _h_zs, sizeof(Table), cudaMemcpyHostToDevice));
            
            size_t cap_after = allocator.gpu_capacity();
            assert(cap_before > cap_after);
            size_t alloced = cap_before - cap_after;
            SYNCALL;
            LOGENDING(1, 4, "(reserved %zd MB, %zd partitions).", ratio(alloced, MB), _num_partitions);
            assert(_num_partitions);
            return _num_partitions;
        }

        // Doesn't reallocate memory.
        size_t resize(const size_t& num_qubits, const size_t& max_window_bytes, const bool& prefix, const bool& measuring, const bool& alloc_signs, const size_t& forced_num_partitions = 0) {
            if (!num_qubits)
                LOGERROR("cannot resize tableau for 0 qubits.");
            if (_num_qubits < num_qubits)
                LOGERROR("not enough memory for tableau resizing.");
            LOGN2(1, "Resizing tableau for %s%lld qubits%s.. ", CREPORTVAL, int64(num_qubits), CNORMAL);
            assert(_num_qubits >= num_qubits);
            // Reset the tableau.
            reset();
            size_t sign_word_size = sizeof(sign_t);
            size_t cap_before = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
            // Partition the tableau if needed.
            _num_qubits = num_qubits;
            const size_t num_words_major_whole_tableau = get_num_words(_num_qubits);
            _num_qubits_padded = get_num_padded_bits(num_qubits);
            _num_words_major = num_words_major_whole_tableau;
            if (!prefix && alloc_signs) _num_sign_words = _num_words_major;
            _num_words = _num_words_major * _num_qubits_padded;
            const size_t max_padded_bits_two_tables = 2 * _num_qubits_padded;
            size_t expected_capacity_required = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
            _num_partitions = 1;
            assert(_num_words_major * max_padded_bits_two_tables == 2 * _num_words);
            while ((forced_num_partitions && forced_num_partitions > _num_partitions) || (expected_capacity_required >= cap_before && _num_words_major > 1)) {
                _num_words_major = (_num_words_major + 0.5) / 1.5;
                if (!prefix && alloc_signs) _num_sign_words = _num_words_major;
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
            if (!prefix && alloc_signs) _num_sign_words = _num_words_major;
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
            if (!prefix && alloc_signs) {     
                assert(_ss_data != nullptr);
                _h_ss->alloc(_ss_data, _num_qubits_padded, _num_sign_words, false);
                CHECK(cudaMemcpyAsync(_ss, _h_ss, sizeof(Signs), cudaMemcpyHostToDevice));
            }
            CHECK(cudaMemcpyAsync(_xs, _h_xs, sizeof(Table), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(_zs, _h_zs, sizeof(Table), cudaMemcpyHostToDevice));
            
            size_t cap_after = 2 * _num_words * sizeof(word_std_t) + _num_sign_words * sign_word_size + max_window_bytes;
            assert(cap_before >= cap_after);
            size_t alloced = cap_before - cap_after;
            SYNCALL;
            LOGENDING(1, 4, "(reserved %zd MB, %zd partitions).", ratio(alloced, MB), _num_partitions);
            assert(_num_partitions);
            return _num_partitions;
        }

        void reset_signs() const {
            if (_ss_data != nullptr)
                CHECK(cudaMemsetAsync(_ss_data, 0, _num_sign_words * sizeof(sign_t)));
        }

        void reset() const {
            assert(_xs_data != nullptr);
            assert(_zs_data != nullptr);
            reset_signs();
            CHECK(cudaMemsetAsync(_xs_data, 0, _num_words * sizeof(word_t)));
            CHECK(cudaMemsetAsync(_zs_data, 0, _num_words * sizeof(word_t)));
        }

        inline size_t size() const { return 2 * _num_words + _num_sign_words; }

        inline size_t num_qubits() const { return _num_qubits; }

        inline size_t num_words_per_table() const { return _num_words; }

        inline size_t num_qubits_padded() const { return _num_qubits_padded; }

        inline size_t num_words_major() const { return _num_words_major; }

        inline size_t num_words_minor() const { return _num_words_minor; }

        inline Signs* signs() const { assert(_ss != nullptr); return _ss; }

        inline Table* xtable() const { assert(_xs != nullptr); return _xs; }

        inline Table* ztable() const { assert(_zs != nullptr); return _zs; }

        inline sign_t* sdata() { assert(_ss_data != nullptr); return _ss_data; }

        inline word_std_t* xdata(const size_t& offset = 0) { 
            assert(_xs_data != nullptr);
            assert(offset < _num_words);
            return reinterpret_cast<word_std_t*>(_xs_data) + offset;
        }

        inline word_std_t* zdata(const size_t& offset = 0) { 
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

        void copy_to_host(Table* h_xs, Table* h_zs, Signs* h_ss = nullptr) const {
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
                h_ss->alloc_host(_num_qubits_padded, _num_sign_words);
                CHECK(cudaMemcpy(h_ss->data(), _ss_data, sizeof(sign_t) * _num_sign_words, cudaMemcpyDeviceToHost));
            }
        }

    };

} 
