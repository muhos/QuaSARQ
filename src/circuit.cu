#include "circuit.cuh"
#include "noise.cuh"

namespace QuaSARQ {

    void DeviceCircuit::initiate(const size_t& max_qubits, const size_t& max_references, const size_t& max_buckets) {
        if (!max_qubits || max_qubits > MAX_QUBITS)
            LOGERROR("maximum number of qubits %lld per window is invalid.", int64(max_qubits));
        if (!max_references || max_references > MAX_QUBITS)
            LOGERROR("maximum number of references %lld per window is invalid.", int64(max_references));
        if (!max_buckets || max_buckets > NO_REF)
            LOGERROR("maximum number of buckets %lld per window is invalid.", int64(max_buckets));		
        if (this->max_references < max_references) {
            LOGN2(2, "Resizing a (pinned) window for %lld references.. ", int64(max_references));
            this->max_references = max_references;
            _references = allocator.allocate<gate_ref_t>(max_references, Region::Stable);
            allocator.resize_pinned<gate_ref_t>(_pinned_references, max_references);
            LOGDONE(2, 4);
        }
        if (this->max_buckets < max_buckets) {
            LOGN2(2, "Resizing a (pinned) window for %lld buckets.. ", int64(max_buckets));
            this->max_buckets = max_buckets;
            _buckets = allocator.allocate<bucket_t>(max_buckets, Region::Stable);
            allocator.resize_pinned<bucket_t>(_pinned_buckets, max_buckets);
            LOGDONE(2, 4);
        }
        this->max_qubits = max_qubits;
    }

    void DeviceCircuit::init_noise_states(const uint64& seed, const size_t& max_gates, const cudaStream_t& stream) {
        if (_max_noise_gates < max_gates) {
            _max_noise_gates = max_gates;
            _noise_states = allocator.allocate<curand_algorithm_t>(max_gates, Region::Stable);
            _noise_paulis = allocator.allocate<uint32>(max_gates, Region::Stable);
        }
        if (_noise_states == nullptr || _noise_paulis == nullptr)
            LOGERROR("failed to allocate noise state buffers.");
        dim3 block(256), grid;
        OPTIMIZEBLOCKS(grid.x, max_gates, block.x);
        setup_noise_k<<<grid, block, 0, stream>>>(_noise_states, seed, max_gates);
    }

    void DeviceCircuit::copyfrom(	
                                    Statistics& 		stats, 
                                    Circuit& 			circuit, 
                                    const depth_t& 		depth_level, 
                                    const bool& 		reversed, 
                                    const bool& 		sync, 
                                    const cudaStream_t& s1, 
                                    const cudaStream_t& s2) {
        if (_references == nullptr)
            LOGERROR("cannot copy empty references to device.");
        if (_buckets == nullptr)
            LOGERROR("cannot copy empty gates to device.");
        if (buckets_offset >= circuit.num_buckets()) 
            LOGERROR("buckets offset overflow during gates transfer to GPU.");
        if (reversed) {
            circuit.dagger(depth_level);
        }
        num_gates = circuit[depth_level].size();
        assert(num_gates <= max_qubits);
        const auto curr_num_buckets = circuit.num_buckets(depth_level);
        assert(num_gates <= max_references);
        assert(curr_num_buckets <= max_buckets);
        const auto* window = circuit[depth_level].data();
        const auto* buckets = circuit.data(buckets_offset);
        double ttime = 0;
        if (sync) cutimer.start(s1);
        LOGN2(2, "Copying %lld references and %lld buckets (offset by %c%lld) per depth level %lld %ssynchroneously.. ", 
            int64(num_gates), 
            int64(curr_num_buckets), 
            reversed ? '-' : '+' , 
            int64(buckets_offset), 
            int64(depth_level), 
            sync ? "" : "a");
        copyhost(_pinned_references, window, num_gates, buckets_offset);
        CHECK(cudaMemcpyAsync(_references, _pinned_references, sizeof(gate_ref_t) * num_gates, cudaMemcpyHostToDevice, s1));
        if (sync) { 
            cutimer.stop(s1); 
            ttime += cutimer.elapsed();
            cutimer.start(s2);
        }
        copyhost(_pinned_buckets, buckets, curr_num_buckets, bucket_t(0));
        CHECK(cudaMemcpyAsync(_buckets, _pinned_buckets, BUCKETSIZE * curr_num_buckets, cudaMemcpyHostToDevice, s2));
        if (sync) {
            cutimer.stop(s2);
            ttime += cutimer.elapsed();
            stats.time.transfer += ttime;
            LOG2(2, "done in %f ms.", ttime);
        }
        if (reversed) {
            const size_t num_buckets_prev = depth_level ? circuit.num_buckets(depth_level - 1) : 0;
            assert(buckets_offset >= num_buckets_prev);
            buckets_offset -= (gate_ref_t) num_buckets_prev;
        }
        else {
            buckets_offset += (gate_ref_t) curr_num_buckets;
        }
        if (!sync) LOGDONE(2, 4);
    }

    void DeviceCircuit::copyto(Circuit& circuit, const depth_t& depth_level) {
        const auto curr_num_buckets = circuit.num_buckets(depth_level);
        const gate_ref_t prev_buckets_offset = buckets_offset - curr_num_buckets;
        if (prev_buckets_offset >= circuit.num_buckets()) 
            LOGERROR("buckets offset overflow during gates transfer to host.");
        LOGN2(2, "Copying back %lld buckets to host per depth level %lld synchroneously.. ", int64(curr_num_buckets), int64(depth_level));
        CHECK(cudaMemcpy(circuit.data(prev_buckets_offset), _buckets, BUCKETSIZE * curr_num_buckets, cudaMemcpyDeviceToHost));
        LOGDONE(2, 4);
    }

}