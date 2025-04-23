#pragma once

#include "definitions.cuh"

namespace QuaSARQ {

	template<class T>
	class SharedMemory {

	public:
		INLINE_DEVICE operator T* () {
			extern __shared__ int _smem[];
			return reinterpret_cast<T*>(_smem);
		}
		INLINE_DEVICE operator const T* () const {
			extern __shared__ int _smem[];
			return reinterpret_cast<T*>(_smem);
		}

	};

}