
#ifndef __CU_DEFINITIONS_H
#define __CU_DEFINITIONS_H  

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "datatypes.hpp"

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

	#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
	#define LASTERR(MESSAGE)	\
		do { \
				cudaError_t ERR = cudaGetLastError(); \
				if (cudaSuccess != ERR) { \
					fprintf(stderr, "\n%s(%i): %s due to (%d) %s\n", __FILE__, __LINE__, MESSAGE, static_cast<int>(ERR), cudaGetErrorString(ERR)); \
					cudaDeviceReset(); \
					exit(1); \
				} \
		} while(0)
	#else
	#define LASTERR(MESSAGE)	do { } while(0)
	#endif

	#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
		#define CHECK(FUNCCALL) \
			do { \
				const cudaError_t returncode = FUNCCALL; \
				if (returncode != cudaSuccess) { \
					fprintf(stderr, "CUDA runtime failure due to %s", cudaGetErrorString(returncode)); \
					cudaDeviceReset(); \
					exit(1); \
				} \
			} while (0)
	#else
		#define CHECK(FUNCCALL) FUNCCALL
	#endif

	#define SYNC(STREAM) CHECK(cudaStreamSynchronize(STREAM))

	#define SYNCALL CHECK(cudaDeviceSynchronize())

	enum Context { CPU, GPU, UNKNOWN };

}

#endif
