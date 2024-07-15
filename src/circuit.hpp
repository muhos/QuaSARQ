

#ifndef __CIRCUIT_H
#define __CIRCUIT_H

#include "vector.hpp"
#include "gate.cuh"

using std::string;
using std::ifstream;

namespace QuaSARQ {

    /**
     * Circuit structure on host.
     * Dynamically and efficiently stores a quantum
     * circuit of any gate type of any number of inputs.  
    */

    typedef uint32 depth_t;
    typedef Vec<bucket_t, size_t> buckets_container;
    typedef Vec<gate_ref_t, size_t> Window;


    constexpr depth_t MAX_DEPTH = -1;
	constexpr size_t GATEBUCKETS = (GATESIZE / BUCKETSIZE);
	#define NBUCKETS(NINPUTS) (GATEBUCKETS + (NINPUTS))

    class Circuit : private buckets_container {

        #define GATE_PTR(REF) ((Gate*) buckets_container::data(REF))

        Vec<Window, depth_t> windows; 
        Vec<size_t, depth_t> nbuckets;

        size_t ngates;

    public:

                    Circuit     () : buckets_container(), ngates(0) 
                    { 
                        // This invariant is a must since we use 
                        // this equation to calculate the number
                        // buckets per gate: size + 3. See NBUCKETS(...).
                        assert(sizeof(qubit_t) == BUCKETSIZE);
                    }
        explicit    Circuit     (const size_t& init_cap) : ngates(0) { 
            assert(sizeof(qubit_t) == BUCKETSIZE);
            buckets_container::reserve(init_cap); 
        }

        inline 
        void        init_depth  (const depth_t& depth) { 
            assert(ngates == 0);
            assert(sizeof(qubit_t) == BUCKETSIZE);
            assert(depth < MAX_DEPTH);
            windows.resize(depth); 
            nbuckets.resize(depth, 0);
        }

        inline 
        bool        empty       () const { return !ngates; } 

        inline
        bucket_t*   data        () { return buckets_container::data(); }

        inline const
        bucket_t*   data        () const { return buckets_container::data(); }

        inline
        bucket_t*   data        (const gate_ref_t& gate_ref) { return buckets_container::data(gate_ref); }

        inline const
        bucket_t*   data        (const gate_ref_t& gate_ref) const { return buckets_container::data(gate_ref); }

        inline
        size_t      num_buckets (const depth_t& depth_level) const { assert(depth_level < MAX_DEPTH); return nbuckets[depth_level]; }

        inline
        size_t      num_buckets () const { return buckets_container::size(); }

        inline
        depth_t     depth       () const { return windows.size(); }

        // Return total number of gates in a cricuit.
        inline
        size_t      num_gates   () const { return ngates; }

        // Return number of gates per window of qubit_t type 
        // since number of gates cannot exceed number of qubits.
        inline
        size_t     num_gates    (const depth_t& depth_level) const {
            assert(depth_level < MAX_DEPTH);
            return windows[depth_level].size();
        }

        inline
        size_t     capacity     () const {
            return num_gates() * sizeof(gate_ref_t) + num_buckets() * BUCKETSIZE;
        }

        inline
        Window&     operator[]  (const depth_t& depth_level) { 
            assert(depth_level < MAX_DEPTH);
            return windows[depth_level];
        }

        inline const
        Window&     operator[]  (const depth_t& depth_level) const { 
            assert(depth_level < MAX_DEPTH);
            return windows[depth_level];
        }

        inline 
        Gate&       gate        (const gate_ref_t& gate_ref) {
            assert(gate_ref < NO_REF);
            return (Gate&)buckets_container::operator[](gate_ref); 
        }

        inline const 
        Gate&       gate        (const gate_ref_t& gate_ref) const { 
            assert(gate_ref < NO_REF);
            return (Gate&)buckets_container::operator[](gate_ref);
        }

        inline
        gate_ref_t& reference   (const depth_t& depth_level, const size_t& gate_index) {
            assert(depth_level < MAX_DEPTH);
            assert(gate_index < MAX_QUBITS);
            return windows[depth_level][gate_index];
        }

        inline const
        gate_ref_t& reference   (const depth_t& depth_level, const size_t& gate_index) const {
            assert(depth_level < MAX_DEPTH);
            assert(gate_index < MAX_QUBITS);
            return windows[depth_level][gate_index];
        }

        inline 
        Gate&       gate        (const depth_t& depth_level, const size_t& gate_index) {
            assert(depth_level < MAX_DEPTH);
            assert(gate_index < MAX_QUBITS);
            return (Gate&)buckets_container::operator[](windows[depth_level][gate_index]); 
        }

        inline const 
        Gate&       gate        (const depth_t& depth_level, const size_t& gate_index) const {
            assert(depth_level < MAX_DEPTH);
            assert(gate_index < MAX_QUBITS);
            return (Gate&)buckets_container::operator[](windows[depth_level][gate_index]);
        }

        inline 
        Gate*       gateptr     (const depth_t& depth_level, const size_t& gate_index) {
            assert(depth_level < MAX_DEPTH);
            assert(gate_index < MAX_QUBITS);
            return GATE_PTR(windows[depth_level][gate_index]); 
        }

        inline const 
        Gate*       gateptr     (const depth_t& depth_level, const size_t& gate_index) const {
            assert(depth_level < MAX_DEPTH);
            assert(gate_index < MAX_QUBITS);
            return GATE_PTR(windows[depth_level][gate_index]); 
        }
        
        inline 
        Gate*       gateptr     (const gate_ref_t& gate_ref) {
            assert(gate_ref < NO_REF);
            return GATE_PTR(gate_ref); 
        }

        inline const 
        Gate*       gateptr     (const gate_ref_t& gate_ref) const {
            assert(gate_ref < NO_REF);
            return GATE_PTR(gate_ref); 
        }

        inline 
        Gate*       addGate     (const depth_t& depth_level, const byte_t& type, const Gate_inputs& inputs) {
			assert(depth_level < MAX_DEPTH);
			const input_size_t size = inputs.size();
			gate_ref_t r = (gate_ref_t) buckets_container::alloc(NBUCKETS(size));
			Gate* gate = new (GATE_PTR(r)) Gate(size);
			assert(gate->size == size);
			assert(gate->capacity() == NBUCKETS(size) * sizeof(bucket_t));
			gate->type = type;
			for (input_size_t i = 0; i < size; i++)
				gate->wires[i] = inputs[i];
            if (windows.size() <= depth_level) {
                windows.expand(depth_level + 1);
                nbuckets.expand(depth_level + 1, 0);
            }
			windows[depth_level].push(r);
			nbuckets[depth_level] += NBUCKETS(size);
			assert(nbuckets[depth_level] <= num_buckets());
			++ngates;
			if (options.verbose > 1) {
				LOGN2(2, "  Gate(d: %d, r = %d):", depth_level, r);
				gate->print();
			}
			return gate;
        }

        inline
            Gate* addGate       (const depth_t& depth_level, const byte_t& type, const input_size_t& size, const qubit_t& c, const qubit_t& t) {
            assert(depth_level < MAX_DEPTH);
            const size_t buckets = NBUCKETS(size);
            gate_ref_t r = (gate_ref_t)buckets_container::alloc(buckets);
            Gate* gate = GATE_PTR(r);
            gate->size = size;
            assert(gate->capacity() == buckets * sizeof(bucket_t));
            gate->type = type;
            gate->wires[0] = c;
            if (c != t) {
                assert(size > 1);
                gate->wires[1] = t;
            }
            if (windows.size() <= depth_level) {
                windows.expand(depth_level + 1);
                nbuckets.expand(depth_level + 1, 0);
            }
            windows[depth_level].push(r);
            nbuckets[depth_level] += buckets;
            assert(nbuckets[depth_level] <= num_buckets());
            ++ngates;
            if (options.verbose > 1) {
                LOGN2(2, "  Gate(d: %d, r = %d):", depth_level, r);
                gate->print();
            }
            return gate;
        }

        inline
        void        copyTo      (Circuit& new_circuit) { 
            new_circuit.copyFrom(*this);
            new_circuit.nbuckets.copyFrom(nbuckets);
            new_circuit.windows.resize(windows.size());
            for (depth_t d = 0; d < windows.size(); d++) {
                new_circuit.windows[d].copyFrom(windows[d]);             
            }
            new_circuit.ngates = ngates;
        }

        inline
        void        destroy     () { 
            clear(true);
            windows.clear(true);
            nbuckets.clear(true);
            ngates = 0;
        }
    };

}

#endif