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

	using DeviceAllocator = cuarena::DeviceArena;

}
