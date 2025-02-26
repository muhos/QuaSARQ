#include "simulator.hpp"
#include "tuner.cuh"

namespace QuaSARQ {

	TableauState ts;
	constexpr bool PRINT_PROGRESS_2D = 0;
	constexpr size_t NSAMPLES = 2;
	constexpr int MIN_PRECISION_HITS = 2;
	constexpr size_t TRIALS = size_t(1e3);
	constexpr double PRECISION = 0.005;
#if	defined(_DEBUG) || defined(DEBUG) || !defined(NDEBUG)
	int64 maxThreadsPerBlock = 256;
	int64 maxThreadsPerBlockY = 32;
	int64 maxThreadsPerBlockX = 64;
#else
	int64 maxThreadsPerBlock = 1024;
	int64 maxThreadsPerBlockY = 512;
	int64 maxThreadsPerBlockX = 512;
#endif
	int64 initThreadsPerBlock1D = 2;
	int64 initThreadsPerBlockX = 2;
	int64 initThreadsPerBlockY = 2;

	#define CONFIG2STRING(CONFIG) \
		if (options.tune_ ## CONFIG) { \
			config += " "; \
			config += #CONFIG; \
			config += " "; \
			config += std::to_string(bestgrid ## CONFIG.x) + " " + std::to_string(bestgrid ## CONFIG.y); \
			config += " "; \
			config += std::to_string(bestblock ## CONFIG.x) + " " + std::to_string(bestblock ## CONFIG.y); \
		}

	void Tuner::write() {
		string config = std::to_string(num_qubits);
		FOREACH_CONFIG(CONFIG2STRING);
		config += "\n";
		fwrite(config.c_str(), 1, config.size(), config_file);
	}

	void Tuner::reset() {
		// Force kernel tuner to run on all kernels.
		FOREACH_CONFIG(CONFIG2RESET);
		// Reset stats.
		stats.reset();
		// Resize tableaux.
		if (num_qubits < tableau.num_qubits()) {
			tableau.resize(num_qubits, winfo.max_window_bytes, measuring);
		}
	}

	void Tuner::run() {
		if (!open_config("wb"))
			LOGERROR("cannot tune without opening a configuration file");
		// Create a tableau in GPU memory for the maximum qubits.
		const size_t max_num_qubits = num_qubits;
		num_partitions = 1;
		tableau.alloc(max_num_qubits, winfo.max_window_bytes, false, measuring);
		gpu_circuit.initiate(num_qubits, winfo.max_parallel_gates, winfo.max_parallel_gates_buckets);
		// Start tuning simulation with max qubits.
		do {
			LOG2(1, "Tuning all kernels for %s%zd qubits%s, %zd partition..", CREPORTVAL, num_qubits, CNORMAL, num_partitions);
			LOG2(1, "");
			// Reset old configuration.
			reset();
			// Parse a circuit.
			parse();
			// Tune identity.
			identity(tableau, 0, num_qubits, custreams, options.initialstate);
			// Start step-wise simulation.
			simulate(0, false);
			// Write configurations.
			write();
			// Clean old circuit.
			circuit.destroy();
			if (!circuit_path.empty()) 
				break;
			// Decrease qubits.
			num_qubits = num_qubits >= options.tuner_step_qubits ? num_qubits - options.tuner_step_qubits : 0;
		} while (num_qubits >= options.tuner_initial_qubits);
		close_config();
		report();
	}

	// Benchmark a 'kernel' up to NSAMPLES times and record the time
	// in AVGTIME per ms. Variables grid and block are assumed.
	#define BENCHMARK_KERNEL(AVGTIME, NSAMPLES, SHAREDSIZE, ...) \
	do { \
		double runtime = 0; \
		for (size_t sample = 0; sample < NSAMPLES; sample++) { \
			if (ts.recover) ts.save_state(); \
			cutimer.start(); \
			kernel <<< grid, block, SHAREDSIZE >>> ( __VA_ARGS__ ); \
			LASTERR("failed to launch kernel for benchmarking"); \
			cutimer.stop(); \
			runtime += cutimer.time(); \
			if (ts.recover) ts.recover_state(); \
		} \
		AVGTIME = (runtime / NSAMPLES); \
	} while(0)

	// Given TIME and MIN, update BESTGRID and BESTBLOCK.
	// Assume block and grid are defined.
	#define BEST_CONFIG(TIME, MIN, BESTGRID, BESTBLOCK, BAILOUT) \
	do { \
		if (TIME < MIN) { \
			if ((MIN - TIME) <= PRECISION && !--min_precision_hits) { \
				LOG2(1, " Found slightly better GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, TIME); \
				BAILOUT = true; \
			} \
			MIN = TIME; \
			BESTBLOCK = block; \
			BESTGRID = grid; \
			if (!BAILOUT) LOG2(1, " Found better GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, TIME); \
		} \
	} while(0)

	#define TUNE_1D(...) \
	do { \
		if (bestBlock.x > 1 || bestGrid.x > 1) { \
			LOG2(2, "\nBest configuration: block(%d), grid(%d) will be used without tuning.", bestBlock.x, bestGrid.x); \
		} \
		else { \
			LOG0(""); \
			LOG2(1, "Tunning %s kernel with maximum of %zd trials and %-.5f milliseconds precision...", opname, TRIALS, PRECISION); \
			int min_precision_hits = MIN_PRECISION_HITS;  \
			const int64 maxBlocksPerGrid = maxGPUBlocks << 1; \
			int64 initBlocksPerGrid = 0; \
			OPTIMIZEBLOCKS(initBlocksPerGrid, size, maxThreadsPerBlock); \
			double minRuntime = double(UINTMAX_MAX); \
			bool early_exit = false; \
			size_t trials = 0; \
			for (int64 gridX = initBlocksPerGrid; gridX <= maxBlocksPerGrid && !early_exit && trials < TRIALS; gridX += 4, trials++) { \
				for (int64 blockX = maxThreadsPerBlock; blockX >= initThreadsPerBlock1D && !early_exit && trials < TRIALS; blockX >>= 1, trials++) { \
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
		} \
	} while(0)

	#define TUNE_1D_FIX_X(...) \
	do { \
		if (bestBlock.x > 1 || bestGrid.x > 1) { \
			LOG2(2, "\nBest configuration: block(%d), grid(%d) will be used without tuning.", bestBlock.x, bestGrid.x); \
		} \
		else { \
			LOG0(""); \
			LOG2(1, "Tunning %s kernel with maximum of %zd trials and %-.5f milliseconds precision...", opname, TRIALS, PRECISION); \
			int min_precision_hits = MIN_PRECISION_HITS;  \
			const int64 maxBlocksPerGrid = maxGPUBlocks << 3; \
			double minRuntime = double(UINTMAX_MAX); \
			bool early_exit = false; \
			size_t trials = 0; \
			for (int64 blockX = maxThreadsPerBlock; blockX >= initThreadsPerBlock1D && !early_exit && trials < TRIALS; blockX >>= 1, trials++) { \
				const size_t shared_size = shared_element_bytes * blockX; \
				if (shared_size > maxGPUSharedMem) continue; \
				if (blockX % maxWarpSize != 0) continue; \
				const int64 gridX = ROUNDUP(size, blockX); \
				if (gridX > maxBlocksPerGrid) continue; \
				dim3 block((uint32)blockX); \
				dim3 grid((uint32)gridX); \
				double avgRuntime = 0; \
				BENCHMARK_KERNEL(avgRuntime, NSAMPLES, shared_size, ## __VA_ARGS__); \
				BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
			} \
			LOG0(""); \
			LOG2(1, "Best GPU time for %s operation using block(%d, 1), and grid(%d, 1): %f ms", opname, bestBlock.x, bestGrid.x, minRuntime); \
			LOG0(""); \
			fflush(stdout); \
		} \
	} while(0)

	#define TUNE_2D(...) \
	do { \
		if (bestBlock.x > 1 || bestGrid.x > 1 || bestBlock.y > 1 || bestGrid.y > 1) { \
			LOG2(2, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
		} \
		else { \
			LOG0(""); \
			LOG2(1, "Tunning %s kernel with maximum of %zd trials and %-.5f milliseconds precision...", opname, TRIALS, PRECISION); \
			int min_precision_hits = MIN_PRECISION_HITS; \
			const bool x_warped = (bool) hasstr(opname, "warped"); \
			int64 initBlocksPerGridX = 0, initBlocksPerGridY = 0; \
			OPTIMIZEBLOCKS2D(initBlocksPerGridY, data_size_in_y, maxThreadsPerBlockY); \
			OPTIMIZEBLOCKS2D(initBlocksPerGridX, data_size_in_x, maxThreadsPerBlockX); \
			double minRuntime = double(UINTMAX_MAX); \
			bool early_exit = false; \
			size_t trials = 0; \
			initBlocksPerGridY = (int64) ceil(initBlocksPerGridY / 1.0); \
			initBlocksPerGridX = (int64) ceil(initBlocksPerGridX / 1.0); \
			const int64 maxBlocksPerGridY = maxGPUBlocks2D << 1; \
			const int64 maxBlocksPerGridX = maxGPUBlocks2D << 1; \
			for (int64 blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 8, trials++) { \
				for (int64 blocksX = initBlocksPerGridX; (blocksX <= maxBlocksPerGridX) && !early_exit && trials < TRIALS; blocksX += 8, trials++) { \
					for (int64 threadsY = maxThreadsPerBlockY; (threadsY >= initThreadsPerBlockY) && !early_exit && trials < TRIALS; threadsY >>= 1) { \
						for (int64 threadsX = maxThreadsPerBlockX; (threadsX >= initThreadsPerBlockX) && !early_exit && trials < TRIALS; threadsX >>= 1) { \
							const int64 threadsPerBlock = threadsX * threadsY; \
							const size_t extended_shared_size = shared_size_yextend ? shared_element_bytes * threadsPerBlock : shared_element_bytes * threadsX; \
							if (x_warped && threadsX > maxWarpSize) continue; \
							if (extended_shared_size > maxGPUSharedMem || threadsPerBlock > maxThreadsPerBlock) continue; \
							/* Avoid deadloack due to warp divergence. */ \
							if (threadsPerBlock % maxWarpSize != 0) continue; \
							dim3 block((uint32)threadsX, (uint32)threadsY); \
							dim3 grid((uint32)blocksX, (uint32)blocksY); \
							double avgRuntime = 0; \
							BENCHMARK_KERNEL(avgRuntime, NSAMPLES, extended_shared_size, ## __VA_ARGS__); \
							if (PRINT_PROGRESS_2D) LOG2(1, "  GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, avgRuntime); fflush(stdout); fflush(stderr); \
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
		} \
	} while(0)

	#define TUNE_2D_FIX_BLOCK_X(GRID_Z, ...) \
	do { \
		if (bestBlock.x > 1 || bestGrid.x > 1 || bestBlock.y > 1 || bestGrid.y > 1) { \
			LOG2(2, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
		} \
		else { \
			LOG0(""); \
			LOG2(1, "Tunning %s kernel with maximum of %zd trials and %-.5f milliseconds precision...", opname, TRIALS, PRECISION); \
			int min_precision_hits = MIN_PRECISION_HITS; \
			int64 initBlocksPerGridX = 0, initBlocksPerGridY = 0; \
			OPTIMIZEBLOCKS2D(initBlocksPerGridY, data_size_in_y, maxThreadsPerBlockY); \
			OPTIMIZEBLOCKS2D(initBlocksPerGridX, data_size_in_x, maxThreadsPerBlockX); \
			double minRuntime = double(UINTMAX_MAX); \
			bool early_exit = false; \
			size_t trials = 0; \
			initBlocksPerGridY = (int64) ceil(initBlocksPerGridY / 1.0); \
			initBlocksPerGridX = (int64) ceil(initBlocksPerGridX / 1.0); \
			const int64 maxBlocksPerGridY = maxGPUBlocks2D << 1; \
			const int64 maxBlocksPerGridX = maxGPUBlocks2D << 1; \
			for (int64 blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 8, trials++) { \
				for (int64 blocksX = initBlocksPerGridX; (blocksX <= maxBlocksPerGridX) && !early_exit && trials < TRIALS; blocksX += 8, trials++) { \
					for (int64 threadsY = maxThreadsPerBlockY; (threadsY >= initThreadsPerBlockY) && !early_exit && trials < TRIALS; threadsY >>= 1) { \
						const int64 threadsPerBlock = threadsX * threadsY; \
						const size_t extended_shared_size = shared_size_yextend ? shared_element_bytes * threadsPerBlock : shared_element_bytes * threadsX; \
						if (extended_shared_size > maxGPUSharedMem || threadsPerBlock > maxThreadsPerBlock) continue; \
						/* Avoid deadloack due to warp divergence. */ \
						if (threadsPerBlock % maxWarpSize != 0) continue; \
						dim3 block((uint32)threadsX, (uint32)threadsY); \
						dim3 grid((uint32)blocksX, (uint32)blocksY, (uint32)GRID_Z); \
						double avgRuntime = 0; \
						BENCHMARK_KERNEL(avgRuntime, NSAMPLES, extended_shared_size, ## __VA_ARGS__); \
						if (PRINT_PROGRESS_2D) LOG2(1, "  GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, avgRuntime); fflush(stdout); fflush(stderr); \
						BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
					} \
				} \
			} \
			LOG2(1, "Best %s configuration found after %zd trials:", opname, trials); \
			LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
			LOG2(1, " Grid  (%-4u, %4u)", bestGrid.x, bestGrid.y); \
			LOG2(1, " Min time: %.4f ms", minRuntime); \
			LOG0(""); \
			fflush(stdout); \
		} \
	} while(0)

	#define TUNE_2D_FIX_GRID_X(...) \
	do { \
		if (bestBlock.x > 1 || bestGrid.x > 1 || bestBlock.y > 1 || bestGrid.y > 1) { \
			LOG2(2, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
		} \
		else { \
			LOG0(""); \
			LOG2(1, "Tunning %s kernel with maximum of %zd trials and %-.5f milliseconds precision...", opname, TRIALS, PRECISION); \
			int min_precision_hits = MIN_PRECISION_HITS; \
			int64 initBlocksPerGridY = 0; \
			OPTIMIZEBLOCKS2D(initBlocksPerGridY, data_size_in_y, maxThreadsPerBlockY); \
			double minRuntime = double(UINTMAX_MAX); \
			bool early_exit = false; \
			size_t trials = 0; \
			initBlocksPerGridY = (int64) ceil(initBlocksPerGridY / 1.0); \
			const int64 maxBlocksPerGridY = maxGPUBlocks2D << 1; \
			const int64 maxBlocksPerGridX = maxGPUBlocks2D << 3; \
			for (int64 blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 4, trials++) { \
				for (int64 threadsY = maxThreadsPerBlockY; (threadsY >= initThreadsPerBlockY) && !early_exit && trials < TRIALS; threadsY >>= 1) { \
					for (int64 threadsX = maxThreadsPerBlockX; (threadsX >= initThreadsPerBlockX) && !early_exit && trials < TRIALS; threadsX >>= 1) { \
						const int64 threadsPerBlock = threadsX * threadsY; \
						const size_t extended_shared_size = shared_size_yextend ? shared_element_bytes * threadsPerBlock : shared_element_bytes * threadsX; \
						if (extended_shared_size >= maxGPUSharedMem || threadsPerBlock > maxThreadsPerBlock) continue; \
						/* Avoid deadloack due to warp divergence. */ \
						if (threadsPerBlock % maxWarpSize != 0) continue; \
						const int64 blocksX = ROUNDUP(data_size_in_x, threadsX); \
						if (blocksX > maxBlocksPerGridX) continue; \
						dim3 block((uint32)threadsX, (uint32)threadsY); \
						dim3 grid((uint32)blocksX, (uint32)blocksY); \
						double avgRuntime = 0; \
						BENCHMARK_KERNEL(avgRuntime, NSAMPLES, extended_shared_size, ## __VA_ARGS__); \
						if (PRINT_PROGRESS_2D) LOG2(1, "  GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, avgRuntime); fflush(stdout); fflush(stderr); \
						BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
					} \
				} \
			} \
			LOG2(1, "Best %s configuration found after %zd trials:", opname, trials); \
			LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
			LOG2(1, " Grid  (%-4u, %4u)", bestGrid.x, bestGrid.y); \
			LOG2(1, " Min time: %.4f ms", minRuntime); \
			LOG0(""); \
			fflush(stdout); \
		} \
	} while(0)

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

	void tune_kernel(void (*kernel)(ConstRefsPointer, ConstBucketsPointer, const size_t, const size_t, 
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
		ConstRefsPointer gate_refs, ConstBucketsPointer gate_buckets, 
		#ifdef INTERLEAVE_XZ
		Table* ps, 
		#else
		Table* xs, Table* zs, 
		#endif
		Signs *ss)
	{
		assert(gate_ref_t(data_size_in_x) == data_size_in_x);
		TUNE_2D(gate_refs, gate_buckets, data_size_in_x, data_size_in_y, TUNE_XZ_TABLES, ss);
	}

	// With measurements.

	void tune_kernel_m(void (*kernel)(const size_t, const size_t, Table*, Table*),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		const size_t& offset, const size_t& size, Table* xs, Table* zs)
	{
		TUNE_1D(offset, size, xs, zs);
	}

	void tune_kernel_m(void (*kernel)(Pivot*, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		Pivot* pivots, const size_t size)
	{
		TUNE_1D(pivots, size);
	}

	void tune_kernel_m(void (*kernel)(Pivot*, bucket_t*, ConstRefsPointer, ConstTablePointer, const size_t, const size_t, const size_t),
		const char* opname, 
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor)
	{
		TUNE_2D(pivots, measurements, refs, inv_xs, num_gates, num_qubits, num_words_minor);
	}

	void tune_kernel_m(void (*kernel)(Pivot*, bucket_t*, ConstRefsPointer, ConstTablePointer, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		Pivot* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
        const size_t& gate_index, const size_t& num_qubits, const size_t& num_words_minor)
	{
		const size_t size = num_qubits;
		TUNE_1D(pivots, measurements, refs, inv_xs, gate_index, num_qubits, num_words_minor);
	}

	void tune_outplace_transpose(void (*kernel)(Table*, Table*, Signs*, ConstTablePointer, ConstTablePointer, ConstSignsPointer, const size_t, const size_t, const size_t),
		const char* opname, 
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Table* xs1, Table* zs1, Signs* ss1, 
        ConstTablePointer xs2, ConstTablePointer zs2, ConstSignsPointer ss2, 
        const size_t& num_words_major, const size_t& num_words_minor, const size_t& num_qubits) 
	{
		const int64 _initThreadsPerBlockY = initThreadsPerBlockY;
		initThreadsPerBlockY = 32;
		TUNE_2D(xs1, zs1, ss1, xs2, zs2, ss2, num_words_major, num_words_minor, num_qubits);
		initThreadsPerBlockY = _initThreadsPerBlockY;
	}

	void tune_inplace_transpose(
		void (*transpose_tiles_kernel)(Table*, Table*, const size_t, const size_t, const bool),
		void (*swap_tiles_kernel)(Table*, Table*, const size_t, const size_t),
		dim3& bestBlockTransposeBits, dim3& bestGridTransposeBits,
		dim3& bestBlockTransposeSwap, dim3& bestGridTransposeSwap,
		Table* xs, Table* zs,
        const size_t& num_words_major, const size_t& num_words_minor, const bool& row_major) 
	{
		int64 threadsX = WORD_BITS;
		bool shared_size_yextend = true;
		if (options.tune_transposebits) {
			void (*kernel)(Table*, Table*, const size_t, const size_t, const bool) = transpose_tiles_kernel;
			dim3 bestBlock = bestBlockTransposeBits, bestGrid = bestGridTransposeBits;
			const size_t shared_element_bytes = sizeof(word_std_t);
			const size_t data_size_in_x = num_words_major;
			const size_t data_size_in_y = 1;
			const char* opname = "Transpose-tiles";
			TUNE_2D_FIX_BLOCK_X(2, xs, zs, num_words_major, num_words_minor, row_major);
			bestBlockTransposeBits = bestBlock, bestGridTransposeBits = bestGrid;
		}
		if (options.tune_transposeswap) {
			void (*kernel)(Table*, Table*, const size_t, const size_t) = swap_tiles_kernel;
			dim3 bestBlock = bestBlockTransposeSwap, bestGrid = bestGridTransposeSwap;
			const size_t shared_element_bytes = 2 * sizeof(word_std_t);
			const size_t data_size_in_x = num_words_minor;
			const size_t data_size_in_y = 1;
			const char* opname = "Swap-tiles";
			TUNE_2D_FIX_BLOCK_X(2, xs, zs, num_words_major, num_words_minor);
			bestBlockTransposeSwap = bestBlock, bestGridTransposeSwap = bestGrid;
		}
	}

	void tune_determinate(void (*kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, ConstTablePointer, ConstTablePointer, ConstSignsPointer, const size_t, const size_t, const size_t),
		const char* opname, 
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs,
        ConstTablePointer inv_xs, ConstTablePointer inv_zs, ConstSignsPointer inv_ss, 
        const size_t num_gates, const size_t num_qubits, const size_t num_words_minor)
	{
		const int64 _initThreadsPerBlockX = initThreadsPerBlockX;
		initThreadsPerBlockX = 32;
		TUNE_2D_FIX_GRID_X(pivots, measurements, refs, inv_xs, inv_zs, inv_ss, num_gates, num_qubits, num_words_minor);
		initThreadsPerBlockX = _initThreadsPerBlockX;
	}

	void tune_single_determinate(void (*kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, ConstTablePointer, ConstTablePointer, ConstSignsPointer, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid, const size_t& shared_element_bytes, 
		ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs,
        ConstTablePointer inv_xs, ConstTablePointer inv_zs, ConstSignsPointer inv_ss, 
        const size_t gate_index, const size_t num_qubits, const size_t num_words_minor)
	{
		const size_t size = num_words_minor;
		TUNE_1D_FIX_X(pivots, measurements, refs, inv_xs, inv_zs, inv_ss, gate_index, num_qubits, num_words_minor);
	}

	void tune_indeterminate(
		void (*copy_kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, Table*, Table*, Signs*, const size_t, const size_t, const size_t),
		void (*phase1_kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, Table*, Table*, Signs*, const size_t, const size_t, const size_t),
		void (*phase2_kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, Table*, Table*, Signs*, const size_t, const size_t, const size_t),
		dim3& bestBlockCopy, dim3& bestGridCopy,
		dim3& bestBlockPhase1, dim3& bestGridPhase1,
		dim3& bestBlockPhase2, dim3& bestGridPhase2,
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		ConstPivotsPointer pivots, bucket_t* measurements, ConstRefsPointer refs, 
        Table* inv_xs, Table* inv_zs, Signs *inv_ss,
        const size_t gate_index, const size_t num_qubits, const size_t num_words_minor)
	{
		fflush(stdout), fflush(stderr);
		void (*kernel)(ConstPivotsPointer, bucket_t*, ConstRefsPointer, Table*, Table*, Signs*, const size_t, const size_t, const size_t);
		// Tune the copy kernel.
		if (options.tune_copyindeterminate) {
			dim3 bestBlock = bestBlockCopy, bestGrid = bestGridCopy;
			kernel = copy_kernel;
			size_t size = num_words_minor;
			const char* opname = "Copy";
			TUNE_1D(pivots, measurements, refs, inv_xs, inv_zs, inv_ss, gate_index, num_qubits, num_words_minor);
			bestBlockCopy = bestBlock;
			bestGridCopy = bestGrid;
		}
		// Tune phase1 kernel.
		if (options.tune_phase1indeterminate) {
			dim3 bestBlock = bestBlockPhase1, bestGrid = bestGridPhase1;
			kernel = phase1_kernel;
			const size_t data_size_in_x = num_words_minor;
			const size_t data_size_in_y = 2 * num_qubits;
			const char* opname = "Phase1";
			TUNE_2D(pivots, measurements, refs, inv_xs, inv_zs, inv_ss, gate_index, num_qubits, num_words_minor);
			bestBlockPhase1 = bestBlock;
			bestGridPhase1 = bestGrid;
		}
		// Tune phase2 kernel.
		if (options.tune_phase2indeterminate) {
			dim3 bestBlock = bestBlockPhase2, bestGrid = bestGridPhase2;
			kernel = phase2_kernel;
			size_t size = 2 * num_qubits;
			const char* opname = "Phase2";
			TUNE_1D(pivots, measurements, refs, inv_xs, inv_zs, inv_ss, gate_index, num_qubits, num_words_minor);
			bestBlockPhase2 = bestBlock;
			bestGridPhase2 = bestGrid;
		}
	}

}

