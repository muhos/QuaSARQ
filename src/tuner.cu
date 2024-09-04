#include "simulator.hpp"
#include "tuner.cuh"

namespace QuaSARQ {

	// Even number so that changes can be undone at the last sample.
	constexpr size_t NSAMPLES = 2;
	constexpr size_t TRIALS = size_t(1e3);
	constexpr double PRECISION = 0.001;
	constexpr int MIN_PRECISION_HITS = 2;
#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
	constexpr grid_t maxThreadsPerBlock = 512;
#else
	constexpr grid_t maxThreadsPerBlock = 1024;
#endif
	constexpr grid_t maxThreadsPerBlockY = 512;
	constexpr grid_t maxThreadsPerBlockX = 256;
	constexpr grid_t initThreadsPerBlock = 2;

	Tuner::Tuner() : 
		Simulator() 
	{ }

	void Tuner::write() {
		string config = std::to_string(num_qubits) + " ";
		if (options.tune_identity) {
			config += "I " + std::to_string(bestGridIdentity.x) + " " + std::to_string(bestBlockIdentity.x) + " ";
		}
		if (options.tune_step) {
			config += "S " + std::to_string(bestGridStep.x) + " " + std::to_string(bestGridStep.y) + " ";
			config += std::to_string(bestBlockStep.x) + " " + std::to_string(bestBlockStep.y) + " ";
		}
		config += "\n";
		fwrite(config.c_str(), 1, config.size(), configfile);
	}

	void Tuner::reset() {
		// Force kernel tuner to run on all kernels.
		bestBlockIdentity = dim3();
		bestGridIdentity = dim3();
		bestBlockReset = dim3();
		bestGridReset = dim3();
		bestBlockStep = dim3();
		bestGridStep = dim3();
	}

	void Tuner::run() {
		if (!open_config("w"))
			LOGERROR("cannot tune without opening a configuration file");
		// Create a tableau in GPU memory for the maximum qubits.
		const size_t max_num_qubits = num_qubits;
		num_partitions = tableau.alloc(max_num_qubits, max_window_bytes, measuring);
		if (num_partitions > 1) num_partitions = 1;
		gpu_circuit.initiate(max_parallel_gates, max_parallel_gates_buckets);
		// Start tuning simulation with max qubits.
		do {
			LOGN2(1, "Tuning all kernels for %s%zd qubits%s, %zd partition..", CREPORTVAL, num_qubits, CNORMAL, num_partitions);
			// Reset old configuration.
			reset();
			// Tune identity.
			identity(tableau, 0, tableau.num_words_per_column() * WORD_BITS, custreams, options.initialstate);
			// Parse a circuit.
			parse();
			// Start step-wise simulation.
			simulate(0, false);
			// Write configurations.
			write();
			// Clean old circuit.
			circuit.destroy();
			// Decrease qubits.
			num_qubits = num_qubits >= options.tuner_step_qubits ? num_qubits - options.tuner_step_qubits : 0;

		} while (num_qubits >= options.tuner_initial_qubits);
		close_config();
	}

	// Benchmark a 'kernel' up to NSAMPLES times and record the time
	// in AVGTIME per ms. Variables grid and block are assumed.
#define BENCHMARK_KERNEL(AVGTIME, NSAMPLES, SHAREDSIZE, ...) \
	do { \
		double runtime = 0; \
		for (size_t sample = 0; sample < NSAMPLES; sample++) { \
			cutimer.start(); \
			kernel <<< grid, block, SHAREDSIZE >>> ( __VA_ARGS__ ); \
			LASTERR("failed to launch kernel for benchmarking"); \
			cutimer.stop(); \
			runtime += cutimer.time(); \
		} \
		AVGTIME = (runtime / NSAMPLES); \
	} while(0)

	// Given TIME and MIN, update BESTGRID and BESTBLOCK.
	// Assume block and grid are defined.
#define BEST_CONFIG(TIME, MIN, BESTGRID, BESTBLOCK, BAILOUT) \
	if (TIME < MIN) { \
		if ((MIN - TIME) <= PRECISION && !--min_precision_hits) { \
			LOG2(1, " Found slightly better GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, TIME); \
			BAILOUT = true; \
		} \
		MIN = TIME; \
		BESTBLOCK = block; \
		BESTGRID = grid; \
		if (!BAILOUT) LOG2(1, " Found better GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, TIME); \
	}

#define TUNE_1D(...) \
	{ \
		if (bestBlock.x > 1 || bestGrid.x > 1) { \
			LOG2(2, "\nBest configuration: block(%d), grid(%d) will be used without tuning.", bestBlock.x, bestGrid.x); \
			return; \
		} \
		LOG0(""); \
		LOG2(1, "Tunning %s kernel with maximum of %zd trials and %-.5f milliseconds precision...", opname, TRIALS, PRECISION); \
		int min_precision_hits = MIN_PRECISION_HITS;  \
		const grid_t maxBlocksPerGrid = maxGPUBlocks << 1; \
		OPTIMIZEBLOCKS(initBlocksPerGrid, size, initThreadsPerBlock); \
		double minRuntime = double(UINTMAX_MAX); \
		bool early_exit = false; \
		size_t trials = 0; \
		for (grid_t gridX = initBlocksPerGrid; gridX <= maxBlocksPerGrid && !early_exit && trials < TRIALS; gridX += 4, trials++) { \
			for (grid_t blockX = initThreadsPerBlock; blockX <= maxThreadsPerBlock && !early_exit && trials < TRIALS; blockX <<= 1, trials++) { \
				if (blockX > maxWarpSize && blockX % maxWarpSize != 0) \
					continue; \
				dim3 block((uint32)blockX); \
				dim3 grid((uint32)gridX); \
				double avgRuntime = 0; \
				BENCHMARK_KERNEL(avgRuntime, NSAMPLES, 0, ## __VA_ARGS__); \
				BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
			} \
		} \
		LOG0(""); \
		LOG2(1, "Best GPU time for %s operation using block(%d, 1), and grid(%d, 1): %f ms", opname, bestBlock.x, bestGrid.x, minRuntime); \
		LOG0(""); \
		fflush(stdout); \
	}

#define TUNE_2D(...) \
	{ \
		if (bestBlock.x > 1 || bestGrid.x > 1 || bestBlock.y > 1 || bestGrid.y > 1) { \
			LOG2(2, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
			return; \
		} \
		LOG0(""); \
		LOG2(1, "Tunning %s kernel with maximum of %zd trials and %-.5f milliseconds precision...", opname, TRIALS, PRECISION); \
		int min_precision_hits = MIN_PRECISION_HITS; \
		const bool x_warped = hasstr(opname, "warped"); \
		OPTIMIZEBLOCKS2D(initBlocksPerGridY, data_size_in_y, initThreadsPerBlock); \
		OPTIMIZEBLOCKS2D(initBlocksPerGridX, data_size_in_x, initThreadsPerBlock); \
		double minRuntime = double(UINTMAX_MAX); \
		bool early_exit = false; \
		size_t trials = 0; \
		initBlocksPerGridY = (grid_t) ceil(initBlocksPerGridY / 1.0); \
		initBlocksPerGridX = (grid_t) ceil(initBlocksPerGridX / 1.0); \
		const grid_t maxBlocksPerGridY = maxGPUBlocks2D; \
		const grid_t maxBlocksPerGridX = maxGPUBlocks2D << 1; \
		for (grid_t blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 8, trials++) { \
			for (grid_t blocksX = initBlocksPerGridX; (blocksX <= maxBlocksPerGridX) && !early_exit && trials < TRIALS; blocksX += 8, trials++) { \
				for (grid_t threadsY = initThreadsPerBlock; (threadsY <= maxThreadsPerBlockY) && !early_exit && trials < TRIALS; threadsY <<= 1) { \
					for (grid_t threadsX = initThreadsPerBlock; (threadsX <= maxThreadsPerBlockX) && !early_exit && trials < TRIALS; threadsX <<= 1) { \
						const grid_t threadsPerBlock = threadsX * threadsY; \
						const size_t extended_shared_size = shared_size_yextend ? shared_element_bytes * threadsPerBlock : shared_element_bytes * threadsX; \
						if (x_warped && threadsX > maxWarpSize) continue; \
						if (extended_shared_size > maxGPUSharedMem || threadsPerBlock > maxThreadsPerBlock) continue; \
						/* Avoid deadloack due to warp divergence. */ \
						if ((threadsX > maxWarpSize && threadsX % maxWarpSize != 0) || (threadsY > maxWarpSize && threadsY % maxWarpSize != 0)) \
							continue; \
						dim3 block((uint32)threadsX, (uint32)threadsY); \
						dim3 grid((uint32)blocksX, (uint32)blocksY); \
						double avgRuntime = 0; \
						BENCHMARK_KERNEL(avgRuntime, NSAMPLES, extended_shared_size, ## __VA_ARGS__); \
						BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
					} \
				} \
			} \
		} \
		LOG2(1, "Best %s configuration found after %zd trials:", opname, trials); \
		LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
		LOG2(1, " Grid  (%-4u, %4u)", bestGrid.x, bestGrid.y); \
		LOG2(1, " Min time: %.4f ms", minRuntime); \
		LOG0(""); \
		fflush(stdout); \
	}

	void tune_kernel(void (*kernel)(const size_t, const size_t, Table*),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		const size_t& offset, const size_t& size, Table* ps)
	{
		TUNE_1D(offset, size, ps);
	}

	#ifdef INTERLEAVE_XZ
	#define TUNE_XZ_TABLES ps
	#else
	#define TUNE_XZ_TABLES xs, zs
	#endif

	void tune_kernel(void (*kernel)(const size_t, 
		#ifdef INTERLEAVE_XZ
		Table*,
		#else
		Table*, Table*, 
		#endif
		Signs *),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		const size_t& size, 
		#ifdef INTERLEAVE_XZ
		Table* ps, 
		#else
		Table* xs, Table* zs, 
		#endif
		Signs *ss)
	{
		TUNE_1D(size, TUNE_XZ_TABLES, ss);
	}

	void tune_kernel(void (*kernel)(const gate_ref_t*, const bucket_t*, const size_t, const size_t, 
		#ifdef INTERLEAVE_XZ
		Table*,
		#else
		Table*, Table*, 
		#endif
		Signs *),
		const char* opname,
		dim3& bestBlock, dim3& bestGrid, 
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		const gate_ref_t* gate_refs, const bucket_t* gate_buckets, const size_t& num_gates, const size_t& num_words_per_column, 
		#ifdef INTERLEAVE_XZ
		Table* ps, 
		#else
		Table* xs, Table* zs, 
		#endif
		Signs *ss)
	{
		assert(gate_ref_t(num_gates) == num_gates);
		assert(gate_ref_t(data_size_in_x) == data_size_in_x);
		TUNE_2D(gate_refs, gate_buckets, num_gates, num_words_per_column, TUNE_XZ_TABLES, ss);
	}

}

