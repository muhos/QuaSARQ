#pragma once

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "datatypes.hpp"
#include "error.hpp"

namespace QuaSARQ {

	#define INLINE inline

	#ifdef __GNUC__
	#define __forceinline __attribute__((always_inline))
	#endif

	#if !defined(INLINED_HOST)
	#define INLINED_HOST INLINE
	#endif

	#if !defined(DEVICE)
	#define DEVICE __device__
	#endif

	#if !defined(INLINE_DEVICE)
	#define INLINE_DEVICE INLINE DEVICE
	#endif

	#if !defined(INLINE_ALL)
	#define INLINE_ALL INLINE __host__ DEVICE
	#endif
	
	#if !defined(NOINLINE_DEVICE)
	#define NOINLINE_DEVICE DEVICE
	#endif

	#if !defined(NOINLINE_ALL)
	#define NOINLINE_ALL __host__ DEVICE
	#endif

	#define LASTERR(MESSAGE)	\
		do { \
				cudaError_t ERR = cudaGetLastError(); \
				if (cudaSuccess != ERR) { \
					char quasarq_cuda_error_message[2048]; \
					std::snprintf(quasarq_cuda_error_message, sizeof(quasarq_cuda_error_message), \
						"%s(%i): %s due to (%d) %s", __FILE__, __LINE__, \
						MESSAGE, static_cast<int>(ERR), cudaGetErrorString(ERR)); \
					fprintf(stderr, "\n%s\n", quasarq_cuda_error_message); \
					throw QuaSARQ::fatal_error(quasarq_cuda_error_message); \
				} \
		} while(0)

	#define CHECK(FUNCCALL) \
		do { \
			const cudaError_t returncode = FUNCCALL; \
			if (returncode != cudaSuccess) { \
				char quasarq_cuda_error_message[2048]; \
				std::snprintf(quasarq_cuda_error_message, sizeof(quasarq_cuda_error_message), \
					"%s(%i): CUDA runtime failure due to (%d) %s", __FILE__, __LINE__, \
					static_cast<int>(returncode), cudaGetErrorString(returncode)); \
				fprintf(stderr, "%s\n", quasarq_cuda_error_message); \
				throw QuaSARQ::fatal_error(quasarq_cuda_error_message); \
			} \
		} while (0)

	#define SYNC(STREAM) CHECK(cudaStreamSynchronize(STREAM))

	#define SYNCALL CHECK(cudaDeviceSynchronize())

	#define NUM_COPY_STREAMS 4
	#define NUM_COMPUTE_STREAMS 2

}
