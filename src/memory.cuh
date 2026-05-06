#pragma once

#include "definitions.cuh"
#include "constants.hpp"
#include "logging.hpp"
#include "malloc.hpp"
#include "cuarena/cuarena.cuh"

namespace QuaSARQ {

	struct tableau_memory_error {
		tableau_memory_error() { fflush(stderr); fflush(stdout); }
	};

	using DeviceAllocator = cuArena::DeviceArena;
	using Region = cuArena::Region;

	inline size_t gpu_stable_avail(const DeviceAllocator& alloc) noexcept {
		if (alloc.gpu_stable_capacity() == 0)
			return alloc.gpu_available();
		return alloc.gpu_stable_capacity() - alloc.gpu_stable_used();
	}

}
