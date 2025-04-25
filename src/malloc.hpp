#pragma once

#include "logging.hpp"
#include <cstdlib>
#include <cstring>

namespace QuaSARQ {

	struct CPU_memory_exception {
		CPU_memory_exception() { fflush(stderr); fflush(stdout); }
	};

	template <class T>
	T* malloc(const size_t& numElements) {
		if (!numElements) LOGERROR("caught zero-memory size at %s", __func__);
		T* _mem = (T*)std::malloc(numElements * sizeof(T));
		if (_mem == nullptr) throw CPU_memory_exception();
		return _mem;
	}

	template <class T>
	T* calloc(const size_t& numElements) {
		if (!numElements) LOGERROR("caught zero-memory size at %s", __func__);
		T* _mem = (T*)std::calloc(numElements, sizeof(T));
		if (_mem == nullptr) throw CPU_memory_exception();
		return _mem;
	}

	template <class T>
	void ralloc(T*& mem, const size_t& bytes) {
		if (!bytes) LOGERROR("caught zero-memory size at %s", __func__);
		T* _mem = nullptr;
		_mem = (T*)std::realloc(mem, bytes);
		if (_mem == nullptr) throw CPU_memory_exception();
		mem = _mem;
	}

	template <class T>
	void shrinkAlloc(T*& mem, const size_t& bytes) {
		if (!bytes) LOGERROR("caught zero-memory size at %s", __func__);
		T* _mem = nullptr;
		_mem = (T*)std::realloc(_mem, bytes);
		if (_mem == nullptr) throw CPU_memory_exception();
		std::memcpy(_mem, mem, bytes);
		std::free(mem);
		mem = _mem;
	}

	#define RESERVE(DATATYPE,MAXCAP,MEM,CAP,MINCAP) \
		if (CAP < (MINCAP)) {	\
			CAP = (CAP > ((MAXCAP) - CAP)) ? (MINCAP) : (CAP << 1);	\
			if (CAP < (MINCAP)) CAP = (MINCAP);	\
			ralloc(MEM, sizeof(DATATYPE) * CAP);	\
		}


}