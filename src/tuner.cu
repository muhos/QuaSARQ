#include "simulator.hpp"
#include "tuner.cuh"

namespace QuaSARQ {

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
			if (measuring) {
				prefix.resize(tableau, winfo.max_window_bytes);
			}
		}
	}

	void Tuner::run() {
		if (!open_config("ab"))
			LOGERROR("cannot tune without opening a configuration file");
		// Create a tableau in GPU memory for the maximum qubits.
		const size_t max_num_qubits = num_qubits;
		num_partitions = 1;
		tableau.alloc(max_num_qubits, winfo.max_window_bytes, false, measuring);
		if (measuring) {
			prefix.alloc(tableau, config_qubits, winfo.max_window_bytes);
		}
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
			LOG2(3, "\nBest configuration: block(%d), grid(%d) will be used without tuning.", bestBlock.x, bestGrid.x); \
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
					const size_t shared_size = shared_element_bytes * blockX; \
					if (shared_size > maxGPUSharedMem) continue; \
					dim3 block((uint32)blockX); \
					dim3 grid((uint32)gridX); \
					double avgRuntime = 0; \
					BENCHMARK_KERNEL(avgRuntime, NSAMPLES, shared_size, ## __VA_ARGS__); \
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
			LOG2(3, "\nBest configuration: block(%d), grid(%d) will be used without tuning.", bestBlock.x, bestGrid.x); \
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
			LOG2(3, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
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
			LOG2(3, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
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
			LOG2(3, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
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

	#define TUNE_2D_PREFIX(SKIP_GRID_CHECK, ...) \
	do { \
		if (bestBlock.x > 1 || bestGrid.x > 1 || bestBlock.y > 1 || bestGrid.y > 1) { \
			LOG2(3, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
		} \
		else { \
			if (initThreadsPerBlockX < 2) \
				LOGERROR("block size cannot be less than 2."); \
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
			for (int64 blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 4, trials++) { \
				for (int64 blocksX = initBlocksPerGridX; (blocksX <= maxBlocksPerGridX) && !early_exit && trials < TRIALS; blocksX += 4, trials++) { \
					for (int64 threadsY = maxThreadsPerBlockY; (threadsY >= initThreadsPerBlockY) && !early_exit && trials < TRIALS; threadsY >>= 1) { \
						for (int64 threadsX = MIN_BLOCK_INTERMEDIATE_SIZE; threadsX <= maxThreadsPerBlockX && !early_exit && trials < TRIALS; threadsX <<= 1) { \
							if (nextPow2(threadsX) != threadsX) \
								LOGERROR("non-power-of-2 block size is not allowed."); \
							const size_t pass_1_gridsize = ROUNDUP(data_size_in_x, threadsX); \
							if (!SKIP_GRID_CHECK && pass_1_gridsize > MIN_SINGLE_PASS_THRESHOLD) { \
								if (PRINT_PROGRESS_2D) \
									LOG2(1, "  Skipping block size %lld for data size of %lld", threadsX, data_size_in_x); \
								continue; \
							} \
							const int64 threadsPerBlock = threadsX * threadsY; \
							const size_t shared_size = shared_element_bytes * threadsY * (threadsX + CONFLICT_FREE_OFFSET(threadsX)); \
							if (shared_size > maxGPUSharedMem || threadsPerBlock > maxThreadsPerBlock) continue; \
							/* Avoid deadloack due to warp divergence. */ \
							if (threadsPerBlock % maxWarpSize != 0) continue; \
							dim3 block((uint32)threadsX, (uint32)threadsY); \
							dim3 grid((uint32)blocksX, (uint32)blocksY); \
							double avgRuntime = 0; \
							if (PRINT_PROGRESS_2D) LOG2(1, "  Tuning for block(x:%u, y:%u) and grid(x:%u, y:%u), pass_1_gridsize: %lld", block.x, block.y, grid.x, grid.y, pass_1_gridsize); fflush(stdout); fflush(stderr); \
							BENCHMARK_KERNEL(avgRuntime, NSAMPLES, shared_size, ## __VA_ARGS__); \
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

	#define TUNE_2D_PREFIX_SINGLE(...) \
	do { \
		if (bestBlock.y > 1 || bestGrid.y > 1) { \
			LOG2(3, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
		} \
		else { \
			if (bestBlock.x < 2 || bestBlock.x > MIN_SINGLE_PASS_THRESHOLD) \
				LOGERROR("x-block size is incorrect."); \
			if (bestGrid.x > 1) \
				LOGERROR("x-grid size must be 1."); \
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
			for (int64 blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 4, trials++) { \
				for (int64 threadsY = maxThreadsPerBlockY; (threadsY >= 1) && !early_exit && trials < TRIALS; threadsY >>= 1) { \
					const int64 threadsPerBlock = bestBlock.x * threadsY; \
					const size_t shared_size = shared_element_bytes * threadsY * (bestBlock.x + CONFLICT_FREE_OFFSET(bestBlock.x)); \
					if (shared_size > maxGPUSharedMem || threadsPerBlock > maxThreadsPerBlock) continue; \
					/* Avoid deadloack due to warp divergence. */ \
					if (threadsPerBlock % maxWarpSize != 0) continue; \
					dim3 block((uint32)bestBlock.x, (uint32)threadsY); \
					dim3 grid((uint32)bestGrid.x, (uint32)blocksY); \
					double avgRuntime = 0; \
					if (PRINT_PROGRESS_2D) LOG2(1, "  Tuning for block(x:%u, y:%u) and grid(x:%u, y:%u)", block.x, block.y, grid.x, grid.y); fflush(stdout); fflush(stderr); \
					BENCHMARK_KERNEL(avgRuntime, NSAMPLES, shared_size, ## __VA_ARGS__); \
					BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
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
		size_t shared_element_bytes = 0;
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
		size_t shared_element_bytes = 0;
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
		size_t shared_element_bytes = 0;
		TUNE_1D(offset, size, xs, zs);
	}

	void tune_kernel_m(void (*kernel)(pivot_t*, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		pivot_t* pivots, const size_t size)
	{
		size_t shared_element_bytes = 0;
		TUNE_1D(pivots, size);
	}

	void tune_kernel_m(void (*kernel)(Commutation* commutations, ConstTablePointer, const qubit_t, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		Commutation* commutations, ConstTablePointer inv_xs, const qubit_t qubit, 
		const size_t size, const size_t num_words_major, const size_t num_words_minor)
	{
		size_t shared_element_bytes = 0;
		TUNE_1D(commutations, inv_xs, qubit, size, num_words_major, num_words_minor);
	}

	void tune_kernel_m(void (*kernel)(Table*, Table*, Signs*, const Commutation* commutations, const pivot_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		Table* inv_xs, Table* inv_zs, Signs* ss, const Commutation* commutations, const pivot_t new_pivot, 
		const size_t num_words_major, const size_t num_words_minor) 
	{
		size_t shared_element_bytes = 0;
		size_t size = num_words_minor;
		TUNE_1D(inv_xs, inv_zs, ss, commutations, new_pivot, num_words_major, num_words_minor);
	}

	void tune_kernel_m(void (*kernel)(pivot_t*, bucket_t*, ConstRefsPointer, ConstTablePointer, const size_t, const size_t, const size_t, const size_t),
		const char* opname, 
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const bool& shared_size_yextend,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		pivot_t* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
        const size_t num_gates, const size_t num_qubits, const size_t num_words_major, const size_t num_words_minor)
	{
		TUNE_2D(pivots, measurements, refs, inv_xs, num_gates, num_qubits, num_words_major, num_words_minor);
	}

	void tune_kernel_m(void (*kernel)(Commutation* commutations, pivot_t*, bucket_t*, ConstRefsPointer, ConstTablePointer, const size_t, const size_t, const size_t, const size_t),
		const char* opname, dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		Commutation* commutations, pivot_t* pivots, bucket_t* measurements, ConstRefsPointer refs, ConstTablePointer inv_xs, 
        const size_t& gate_index, const size_t& size, const size_t num_words_major, const size_t num_words_minor)
	{
		TUNE_1D(commutations, pivots, measurements, refs, inv_xs, gate_index, size, num_words_major, num_words_minor);
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

	void tune_prefix_pass_1(
		void (*kernel)(word_std_t*, word_std_t*, word_std_t*, word_std_t*, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		word_std_t* block_intermediate_prefix_z,
		word_std_t* block_intermediate_prefix_x,
		word_std_t* subblocks_prefix_z, 
		word_std_t* subblocks_prefix_x,
		const size_t& num_blocks,
		const size_t& num_words_minor) 
	{
		const char* opname = "prefix pass 1";
		TUNE_2D_PREFIX(
					false,
					block_intermediate_prefix_z, 
                    block_intermediate_prefix_x, 
                    subblocks_prefix_z, 
                    subblocks_prefix_x, 
                    num_blocks, 
                    num_words_minor);
	}

	void tune_inject_pass_1(
		void (*kernel)(Table*, Table*, Table*, Table*, word_std_t *, word_std_t *, 
						const Commutation*, const uint32, const size_t, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Table *prefix_xs, 
        Table *prefix_zs, 
        Table *inv_xs, 
        Table *inv_zs,
        word_std_t *block_intermediate_prefix_z,
        word_std_t *block_intermediate_prefix_x,
		const Commutation* commutations,
		const uint32& pivot,
		const size_t& total_targets,
		const size_t& num_words_major,
		const size_t& num_words_minor)
	{
		const char* opname = "inject pass 1";
		TUNE_2D_PREFIX(
					false,
					prefix_xs, 
        			prefix_zs, 
       				inv_xs, 
        			inv_zs,
        			block_intermediate_prefix_z,
        			block_intermediate_prefix_x,
					commutations,
					pivot,
					total_targets,
					num_words_major,
					num_words_minor);
	}

	void tune_prefix_pass_2(
		void (*kernel)(word_std_t*, word_std_t*, const word_std_t*, const word_std_t*, const size_t, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		word_std_t* block_intermediate_prefix_z,
		word_std_t* block_intermediate_prefix_x,
		const word_std_t* subblocks_prefix_z, 
		const word_std_t* subblocks_prefix_x,
		const size_t& num_blocks,
		const size_t& num_words_minor,
		const size_t& pass_1_blocksize)
	{
		const char* opname = "prefix pass 2";
		const size_t shared_element_bytes = 0;
		const bool shared_size_yextend = false;
		TUNE_2D(block_intermediate_prefix_z, block_intermediate_prefix_x, subblocks_prefix_z, subblocks_prefix_x, num_blocks, num_words_minor, pass_1_blocksize);
	}

	void tune_inject_pass_2(
		void (*kernel)(Table*, Table*, Table*, Table*, const word_std_t *, const word_std_t *, 
						const Commutation*, const uint32, 
						const size_t, const size_t, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Table *prefix_xs, 
        Table *prefix_zs, 
        Table *inv_xs, 
        Table *inv_zs,
        const word_std_t *block_intermediate_prefix_z,
        const word_std_t *block_intermediate_prefix_x,
		const Commutation* commutations,
		const uint32& pivot,
		const size_t& total_targets,
		const size_t& num_words_major,
		const size_t& num_words_minor,
		const size_t& pass_1_blocksize)
	{
		const char* opname = "prefix pass 2";
		const bool shared_size_yextend = true;
		TUNE_2D(
			prefix_xs, 
			prefix_zs, 
			inv_xs, 
			inv_zs,
			block_intermediate_prefix_z,
			block_intermediate_prefix_x,
			commutations,
			pivot,
			total_targets,
			num_words_major,
			num_words_minor,
			pass_1_blocksize
		);
	}

	void tune_collapse_targets(
		void (*kernel)(Table*, Table*, Table*, Table*, Signs *, 
						const Commutation*, const uint32, 
						const size_t, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		Table *prefix_xs, 
        Table *prefix_zs, 
        Table *inv_xs, 
        Table *inv_zs,
		Signs *inv_ss,
		const Commutation* commutations,
		const uint32& pivot,
		const size_t& total_targets,
		const size_t& num_words_major,
		const size_t& num_words_minor)
	{
		const char* opname = "collapse targets";
		const bool shared_size_yextend = true;
		TUNE_2D(
			prefix_xs, 
			prefix_zs, 
			inv_xs, 
			inv_zs,
			inv_ss,
			commutations,
			pivot,
			total_targets,
			num_words_major,
			num_words_minor
		);
	}

	void tune_single_pass(
		void (*kernel)(word_std_t*, word_std_t*, const size_t, const size_t),
		dim3& bestBlock, dim3& bestGrid,
		const size_t& shared_element_bytes, 
		const size_t& data_size_in_x, 
		const size_t& data_size_in_y,
		word_std_t* block_intermediate_prefix_z, 
		word_std_t* block_intermediate_prefix_x,
		const size_t num_chunks,
		const size_t num_words_minor
	)
	{
		const char* opname = "scan single pass";
		TUNE_2D_PREFIX_SINGLE(
			block_intermediate_prefix_z, 
			block_intermediate_prefix_x, 
			num_chunks, 
			num_words_minor
		);
	}


}

