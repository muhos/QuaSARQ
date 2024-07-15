
#ifndef __CU_VECTOR_H
#define __CU_VECTOR_H

#include "definitions.cuh"
#include "datatypes.hpp"
#include "atomic.cuh"
#include "logging.hpp"

namespace QuaSARQ {

	template<typename T>
	class cuVec {
		T* _mem;
		uint32 sz, cap;

		INLINE_DEVICE bool checkAtomicBound(const uint32 idx, const uint32 cap) const
		{
			if (idx >= cap) {
				LOGGPUERROR("vector atomic returned index (%lld) exceeding allocated capacity (%lld)\n", int64(idx), int64(cap));
				return false;
			}
			return true;
		}

		INLINE_DEVICE bool checkBound(const uint32 idx, const uint32 sz) const
		{
			if (idx >= sz) {
				LOGGPUERROR("vector index (%lld) exceeding its size (%lld)\n", int64(idx), int64(sz));
				return false;
			}
			return true;
		}


	public:
		INLINE_ALL					cuVec() : _mem(nullptr), sz(0), cap(0) {}
		INLINE_ALL					~cuVec() { clear(true); }
		INLINE_ALL 		void		alloc(T* head) { _mem = head; }
		INLINE_ALL 		void		alloc(T* head, const uint32& cap) { _mem = head, this->cap = cap; }
		INLINE_DEVICE 	void		alloc(const uint32& cap) { _mem = (T*)(this + 1), this->cap = cap; }
		INLINE_DEVICE 	void		push(const T& val) {
			const uint32 idx = atomicAggInc(&sz);
			assert(checkAtomicBound(idx, cap));
			_mem[idx] = val;
		}
		INLINE_ALL 		void		_pop() { sz--; }
		INLINE_ALL 		void		_shrink(const uint32& n) { sz -= n; }
		INLINE_ALL 		void		_push(const T& val) { _mem[sz++] = val; }
		INLINE_ALL 		cuVec<T>& 	operator=	(cuVec<T>& rhs) { return *this; }
		INLINE_ALL 		const T& 	operator [] (const uint32& idx) const { assert(checkBound(idx, sz)); return _mem[idx]; }
		INLINE_ALL 		T& 			operator [] (const uint32& idx) { assert(checkBound(idx, sz)); return _mem[idx]; }
		INLINE_ALL 		T& 			at(const uint32& idx) { return _mem[idx]; }
		INLINE_ALL 		const T& 	at(const uint32& idx) const { return _mem[idx]; }
		INLINE_ALL		operator T* () { return _mem; }
		INLINE_ALL 		T* 			data() { return _mem; }
		INLINE_ALL 		T* 			end() { return _mem + sz; }
		INLINE_ALL 		T& 			back() { assert(sz); return _mem[sz - 1]; }
		INLINE_ALL 		bool		empty() const { return !sz; }
		INLINE_ALL 		uint32		size() const { return sz; }
		INLINE_ALL 		uint32		capacity() const { return cap; }
		INLINE_ALL 		void		resize(const uint32& n) { assert(n <= cap); sz = n; }
		INLINE_ALL 		void		clear(const bool& _free = false) {
			if (_free) _mem = nullptr, cap = 0;
			sz = 0;
		}
	};

}

#endif