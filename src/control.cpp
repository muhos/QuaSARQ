
#include "control.hpp"
#include "version.hpp"
#include "timer.hpp"
#include "timer.cuh"
#include "grid.cuh"

namespace QuaSARQ {

	Timer timer;
	cuTimer cutimer;
	cudaDeviceProp devProp;
	grid_t maxGPUThreads = 0;
	grid_t maxGPUBlocks = 0;
	grid_t maxGPUBlocks2D = 0;
	grid_t maxWarpSize = 0;
	size_t maxGPUSharedMem = 0;

	size_t sysMemUsed()
	{
		size_t memUsed = 0;
#if defined(__linux__) || defined(__CYGWIN__)
		long rss = 0L;
		FILE *fp = nullptr;
		if ((fp = fopen("/proc/self/statm", "r")) == nullptr)
			return (size_t) 0; 
		if (fscanf(fp, "%*s%ld", &rss) != 1) {
			fclose(fp);
			return (size_t) 0;
		}
		fclose(fp);
		memUsed = (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
#elif defined(_WIN32)
		PROCESS_MEMORY_COUNTERS_EX memInfo;
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&memInfo, sizeof(PROCESS_MEMORY_COUNTERS_EX));
		memUsed = memInfo.WorkingSetSize;
#endif
		return memUsed;
	}

	size_t getAvailSysMem()
	{
#if defined(__linux__) || defined(__CYGWIN__)
		long pages = sysconf(_SC_PHYS_PAGES);
		long page_size = sysconf(_SC_PAGE_SIZE);
		return pages * page_size;
#elif defined(_WIN32)
		MEMORYSTATUSEX memInfo;
		memInfo.dwLength = sizeof(MEMORYSTATUSEX);
		GlobalMemoryStatusEx(&memInfo);
		return memInfo.ullAvailPhys;
#endif
	}

	void set_timeout(int time_limit)
	{
#if defined(__linux__) || defined(__CYGWIN__)
		if (time_limit) {
			rlimit limit;
			getrlimit(RLIMIT_CPU, &limit);
			if (limit.rlim_max == RLIM_INFINITY || (rlim_t)time_limit < limit.rlim_max) {
				limit.rlim_cur = time_limit;
				if (setrlimit(RLIMIT_CPU, &limit) == -1) LOGWARNING("timeout cannot be set");
			}
		}
#elif defined(_WIN32)
		LOGWARNING("timeout not supported on Windows");
#endif
	}

	void set_memoryout(int memory_limit)
	{
#if defined(__linux__)
		if (memory_limit) {
			rlim64_t limitbytes = (rlim64_t)memory_limit * GB;
			rlimit64 limit;
			getrlimit64(RLIMIT_AS, &limit);
			if (limit.rlim_max == RLIM_INFINITY || limitbytes < limit.rlim_max) {
				limit.rlim_cur = limitbytes;
				if (setrlimit64(RLIMIT_AS, &limit) == -1) LOGWARNING("memoryout cannot be set");
			}
		}
#elif defined(__CYGWIN__)
		if (memory_limit) {
			rlim_t limitbytes = (rlim_t)memory_limit * GB;
			rlimit limit;
			getrlimit(RLIMIT_AS, &limit);
			if (limit.rlim_max == RLIM_INFINITY || limitbytes < limit.rlim_max) {
				limit.rlim_cur = limitbytes;
				if (setrlimit(RLIMIT_AS, &limit) == -1) LOGWARNING("memoryout cannot be set");
			}
		}
#elif defined(_WIN32)
		LOGWARNING("memoryout not supported on Windows");
#endif
	}

	void handler_terminate(int)
	{
		fflush(stderr), fflush(stdout);
		LOG1("%s%s%s", CYELLOW, "INTERRUPTED", CNORMAL);
		_exit(EXIT_INTERRUPTED);
	}

	void handler_mercy_interrupt(int)
	{
		fflush(stderr), fflush(stdout);
		LOG1("%s%s%s", CYELLOW, "INTERRUPTED", CNORMAL);
		//sim->interrupt();
	}

	void handler_mercy_timeout(int)
	{
		fflush(stderr), fflush(stdout);
		LOG1("%s%s%s", CYELLOW, "TIME OUT", CNORMAL);
		//sim->interrupt();
	}

	void signal_handler(void h_intr(int), void h_timeout(int))
	{
		signal(SIGINT, h_intr);
		signal(SIGTERM, h_intr);
#ifdef SIGXCPU
		if (h_timeout != nullptr) signal(SIGXCPU, h_timeout);
#endif
	}

	void segmentation_fault(int)
	{
		fflush(stderr), fflush(stdout);
		LOGERRORN("segmentation fault detected.");
		_exit(EXIT_FAILURE);
	}

	void illegal_code(int)
	{
		fflush(stderr), fflush(stdout);
		LOGERRORN("illegal code detected.");
		_exit(EXIT_FAILURE);
	}

	void arithmetic_error(int)
	{
		fflush(stderr), fflush(stdout);
		LOGERRORN("arithmetic flaw detected.");
		_exit(EXIT_FAILURE);
	}

	void getCPUInfo(const int& verbose)
	{
#ifndef __CYGWIN__
		char cpuid[0x40] = { 0 };
#if defined(_WIN32)
		int CPUInfo[4] = { -1 };
		__cpuid(CPUInfo, 0x80000000);
#elif defined(__linux__)
		int CPUInfo[4] = { 0, 0, 0, 0 };
		__cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
#endif
		uint32 nExIds = CPUInfo[0];
		for (uint32 i = 0x80000000; i <= nExIds; ++i) {
#if defined(_WIN32)
			__cpuid(CPUInfo, i);
#elif defined(__linux__)
			__cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
#endif
			if (i == 0x80000002)
				memcpy(cpuid, CPUInfo, sizeof(CPUInfo));
			else if (i == 0x80000003)
				memcpy(cpuid + 16, CPUInfo, sizeof(CPUInfo));
			else if (i == 0x80000004)
				memcpy(cpuid + 32, CPUInfo, sizeof(CPUInfo));
		}
		char * cpu = cpuid;
		while(isSpace(*cpu)) cpu++;
		LOG2(verbose, "Available CPU: %s%s%s", CREPORTVAL, cpu, CNORMAL);
#endif
		size_t _free = getAvailSysMem();
		LOG2(verbose, "Available system memory: %s%zd GB%s", CREPORTVAL, _free / GB, CNORMAL);
		fflush(stdout);
	}

	void getBuildInfo(const int& verbose)
	{
		LOG2(verbose, "Built on %s%s%s at %s%s%s", CREPORTVAL, osystem(), CNORMAL, CREPORTVAL, date(), CNORMAL);
		LOG2(verbose, "      using %s%s %s%s", CREPORTVAL, compiler(), compilemode(), CNORMAL);
		fflush(stdout);
	}

	inline
		int SM2Cores(int major, int minor)
	{
		typedef struct { int SM; int Cores; } SM;

		SM nCores[] = {
			{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
			{0x50, 128}, {0x52, 128}, {0x53, 128},
			{0x60,  64}, {0x61, 128}, {0x62, 128},
			{0x70,  64}, {0x72,  64}, {0x75,  64},
			{0x80,  64}, {0x86, 128}, {0x87, 128}, {0x89, 128},
    		{0x90, 128},
			{-1, -1}
		};

		int index = 0;
		while (nCores[index].SM != -1) {
			if (nCores[index].SM == ((major << 4) + minor)) {
				return nCores[index].Cores;
			}
			index++;
		}
		LOGWARNING("cannot map to cores/SM due to unknown SM");
		return -1;
	}

	int getGPUInfo(const int& verbose)
	{
		int devCount = 0;
		CHECK(cudaGetDeviceCount(&devCount));
		if (!devCount) return 0;
		CHECK(cudaGetDeviceProperties(&devProp, 0));
		assert(devProp.totalGlobalMem);
		if (devProp.warpSize != 32) LOGERROR("GPU warp size not supported");
		size_t _free = devProp.totalGlobalMem;
		size_t _shared_penalty = 512; // enough for the kernel launch
		maxGPUThreads = devProp.multiProcessorCount * devProp.maxThreadsPerMultiProcessor;
		maxGPUBlocks = devProp.multiProcessorCount * devProp.maxBlocksPerMultiProcessor;
		maxGPUBlocks2D = sqrt(maxGPUBlocks);
		maxWarpSize = devProp.warpSize;
		maxGPUSharedMem = devProp.sharedMemPerBlock - _shared_penalty;
		if (!options.quiet_en) {
			LOG2(verbose, "Available GPU: %s%d x %s @ %.2fGHz (compute cap: %d.%d)%s", CREPORTVAL, devCount, devProp.name, ratio((double)devProp.clockRate, 1e6), devProp.major, devProp.minor, CNORMAL);
			const int cores = SM2Cores(devProp.major, devProp.minor);
			LOG2(verbose, "Available GPU Multiprocessors: %s%d MPs (%s cores/MP)%s", CREPORTVAL, devProp.multiProcessorCount, (cores < 0 ? "unknown": std::to_string(cores).c_str()), CNORMAL);
			LOG2(verbose, "Available GPU threads and blocks: %s%lld threads, %lld blocks%s", CREPORTVAL, int64(maxGPUThreads), int64(maxGPUBlocks), CNORMAL);
			LOG2(verbose, "Available Global memory: %s%zd GB%s", CREPORTVAL, _free / GB, CNORMAL);
			fflush(stdout);
		}
		return devCount;
	}

}
