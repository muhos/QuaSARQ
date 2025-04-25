#include "simulator.hpp"
#include "tuner.cuh"
#include "step.cuh"
#include "pivot.cuh"
#include "transpose.cuh"
#include "identity.cuh"
#include "injectswap.cuh"
#include "prefixintra.cuh"

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
	int64 maxThreadsPerBlockY = 1024;
	int64 maxThreadsPerBlockX = 1024;
#endif
	int64 initThreadsPerBlock1D = 2;
	int64 initThreadsPerBlockX = 2;
	int64 initThreadsPerBlockY = 2;

	#define CONFIG2STRING(CONFIG, BLOCKX, BLOCKY, GRIDX, GRIDY) \
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
			tableau.resize(num_qubits, winfo.max_window_bytes, false, measuring, true);
			if (measuring) {
				#if ROW_MAJOR
				inv_tableau.resize(num_qubits, winfo.max_window_bytes, false, measuring, false);
				#endif
				prefix.resize(tableau, winfo.max_window_bytes);
				if (options.check_measurement) {
					mchecker.destroy();
            		mchecker.alloc(num_qubits);
				}
			}
		}
	}

	void Tuner::run() {
		if (!open_config("ab"))
			LOGERROR("cannot tune without opening a configuration file");
		// Create a tableau in GPU memory for the maximum qubits.
		const size_t max_num_qubits = num_qubits;
		num_partitions = 1;
		tableau.alloc(max_num_qubits, winfo.max_window_bytes, false, measuring, true);
		if (measuring) {
			#if ROW_MAJOR
			inv_tableau.alloc(num_qubits, winfo.max_window_bytes, false, measuring, false);
			#endif
			prefix.alloc(tableau, config_qubits, winfo.max_window_bytes);
			pivoting.alloc(num_qubits);
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
			SYNCALL; \
			cutimer.stop(); \
			runtime += cutimer.time(); \
		} \
		AVGTIME = (runtime / NSAMPLES); \
	} while(0)

	#define BENCHMARK_CALL(AVGTIME, NSAMPLES, SHAREDSIZE, CALL, ...) \
	do { \
		double runtime = 0; \
		for (size_t sample = 0; sample < NSAMPLES; sample++) { \
			cutimer.start(); \
			CALL(__VA_ARGS__, block, grid, SHAREDSIZE, 0); \
			LASTERR("failed to launch kernel for benchmarking"); \
			SYNCALL; \
			cutimer.stop(); \
			runtime += cutimer.time(); \
		} \
		AVGTIME = (runtime / NSAMPLES); \
	} while(0)

	#define BENCHMARK_CUB_CALL(AVGTIME, NSAMPLES, CALL, ...) \
	do { \
		double runtime = 0; \
		for (size_t sample = 0; sample < NSAMPLES; sample++) { \
			cutimer.start(); \
			CALL(__VA_ARGS__, block, grid, 0); \
			LASTERR("failed to launch kernel for benchmarking"); \
			SYNCALL; \
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
				LOG2(1, "  Found slightly better GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, TIME); \
				BAILOUT = true; \
			} \
			MIN = TIME; \
			BESTBLOCK = block; \
			BESTGRID = grid; \
			if (!BAILOUT) LOG2(1, "  Found better GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", \
				block.x, block.y, grid.x, grid.y, TIME); \
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
			LOG2(1, " Best GPU time for %s operation using block(%d, 1), and grid(%d, 1): %f ms", opname, bestBlock.x, bestGrid.x, minRuntime); \
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
			LOG2(1, " Best GPU time for %s operation using block(%d, 1), and grid(%d, 1): %f ms", opname, bestBlock.x, bestGrid.x, minRuntime); \
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
			const int64 maxBlocksPerGridX = maxGPUBlocks2D; \
			for (int64 blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 4, trials++) { \
				for (int64 blocksX = initBlocksPerGridX; (blocksX <= maxBlocksPerGridX) && !early_exit && trials < TRIALS; blocksX += 4, trials++) { \
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
			LOG2(1, " Best %s configuration found after %zd trials:", opname, trials); \
			LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
			LOG2(1, " Grid  (%-4u, %4u)", bestGrid.x, bestGrid.y); \
			LOG2(1, " Min time: %.4f ms", minRuntime); \
			fflush(stdout); \
		} \
	} while(0)

	#define TUNE_2D_CALL(CALL, ...) \
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
			const int64 maxBlocksPerGridX = maxGPUBlocks2D; \
			for (int64 blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 4, trials++) { \
				for (int64 blocksX = initBlocksPerGridX; (blocksX <= maxBlocksPerGridX) && !early_exit && trials < TRIALS; blocksX += 4, trials++) { \
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
							BENCHMARK_CALL(avgRuntime, NSAMPLES, extended_shared_size, CALL, ## __VA_ARGS__); \
							if (PRINT_PROGRESS_2D) LOG2(1, "  GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, avgRuntime); fflush(stdout); fflush(stderr); \
							BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
						} \
					} \
				} \
			} \
			LOG2(1, " Best %s configuration found after %zd trials:", opname, trials); \
			LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
			LOG2(1, " Grid  (%-4u, %4u)", bestGrid.x, bestGrid.y); \
			LOG2(1, " Min time: %.4f ms", minRuntime); \
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
			const int64 maxBlocksPerGridY = maxGPUBlocks2D << 2; \
			const int64 maxBlocksPerGridX = maxGPUBlocks2D; \
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
			LOG2(1, " Best %s configuration found after %zd trials:", opname, trials); \
			LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
			LOG2(1, " Grid  (%-4u, %4u)", bestGrid.x, bestGrid.y); \
			LOG2(1, " Min time: %.4f ms", minRuntime); \
			fflush(stdout); \
		} \
	} while(0)

	#define TUNE_2D_CUB(CALL, ...) \
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
			const int64 maxBlocksPerGridY = maxGPUBlocks2D << 2; \
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
							BENCHMARK_CUB_CALL(avgRuntime, NSAMPLES, CALL, ## __VA_ARGS__); \
							if (PRINT_PROGRESS_2D) LOG2(1, "  GPU Time for block(x:%u, y:%u) and grid(x:%u, y:%u): %f ms", block.x, block.y, grid.x, grid.y, avgRuntime); fflush(stdout); fflush(stderr); \
							BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
						} \
					} \
				} \
			} \
			LOG2(1, " Best %s configuration found after %zd trials:", opname, trials); \
			LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
			LOG2(1, " Grid  (%-4u, %4u)", bestGrid.x, bestGrid.y); \
			LOG2(1, " Min time: %.4f ms", minRuntime); \
			fflush(stdout); \
		} \
	} while(0)

	#define TUNE_2D_PREFIX_CUB(SKIP_GRID_CHECK, CALL, ...) \
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
			const int64 maxBlocksPerGridY = maxGPUBlocks2D << 2; \
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
							const size_t shared_size = shared_element_bytes * threadsY * threadsX; \
							if (shared_size > maxGPUSharedMem || threadsPerBlock > maxThreadsPerBlock) continue; \
							/* Avoid deadloack due to warp divergence. */ \
							if (threadsPerBlock % maxWarpSize != 0) continue; \
							dim3 block((uint32)threadsX, (uint32)threadsY); \
							dim3 grid((uint32)blocksX, (uint32)blocksY); \
							double avgRuntime = 0; \
							if (PRINT_PROGRESS_2D) LOG2(1, "  Tuning for block(x:%u, y:%u) and grid(x:%u, y:%u), pass_1_gridsize: %lld", block.x, block.y, grid.x, grid.y, pass_1_gridsize); fflush(stdout); fflush(stderr); \
							BENCHMARK_CUB_CALL(avgRuntime, NSAMPLES, CALL, ## __VA_ARGS__); \
							BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
						} \
					} \
				} \
			} \
			LOG2(1, " Best %s configuration found after %zd trials:", opname, trials); \
			LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
			LOG2(1, " Grid  (%-4u, %4u, %4u)", bestGrid.x, bestGrid.y, bestGrid.z); \
			LOG2(1, " Min time: %.4f ms", minRuntime); \
			fflush(stdout); \
		} \
	} while(0)

	#define TUNE_2D_PREFIX_SINGLE_CUB(CALL, ...) \
	do { \
		if (bestBlock.y > 1 || bestGrid.y > 1) { \
			LOG2(3, "\nBest configuration: block(%d, %d), grid(%d, %d) will be used without tuning.", bestBlock.x, bestBlock.y, bestGrid.x, bestGrid.y); \
		} \
		else { \
			if (bestBlock.x < 2 || bestBlock.x > MIN_SINGLE_PASS_THRESHOLD) \
				LOGERROR("x-block size %d is incorrect.", bestBlock.x); \
			if (bestGrid.x > 1) \
				LOGERROR("x-grid size %d must be 1.", bestGrid.x); \
			LOG0(""); \
			LOG2(1, "Tunning %s kernel with maximum of %zd trials and %-.5f milliseconds precision...", opname, TRIALS, PRECISION); \
			int min_precision_hits = MIN_PRECISION_HITS; \
			int64 initBlocksPerGridY = 0; \
			OPTIMIZEBLOCKS2D(initBlocksPerGridY, data_size_in_y, maxThreadsPerBlockY); \
			double minRuntime = double(UINTMAX_MAX); \
			bool early_exit = false; \
			size_t trials = 0; \
			initBlocksPerGridY = (int64) ceil(initBlocksPerGridY / 1.0); \
			const int64 maxBlocksPerGridY = maxGPUBlocks2D << 2; \
			for (int64 blocksY = initBlocksPerGridY; (blocksY <= maxBlocksPerGridY) && !early_exit && trials < TRIALS; blocksY += 4, trials++) { \
				for (int64 threadsY = maxThreadsPerBlockY; (threadsY >= 1) && !early_exit && trials < TRIALS; threadsY >>= 1) { \
					const int64 threadsPerBlock = bestBlock.x * threadsY; \
					const size_t shared_size = shared_element_bytes * threadsY * bestBlock.x; \
					if (shared_size > maxGPUSharedMem || threadsPerBlock > maxThreadsPerBlock) continue; \
					/* Avoid deadloack due to warp divergence. */ \
					if (threadsPerBlock % maxWarpSize != 0) continue; \
					dim3 block((uint32)bestBlock.x, (uint32)threadsY); \
					dim3 grid((uint32)bestGrid.x, (uint32)blocksY); \
					double avgRuntime = 0; \
					if (PRINT_PROGRESS_2D) LOG2(1, "  Tuning for block(x:%u, y:%u) and grid(x:%u, y:%u)", block.x, block.y, grid.x, grid.y); fflush(stdout); fflush(stderr); \
					BENCHMARK_CUB_CALL(avgRuntime, NSAMPLES, CALL, ## __VA_ARGS__); \
					BEST_CONFIG(avgRuntime, minRuntime, bestGrid, bestBlock, early_exit); \
				} \
			} \
			LOG2(1, " Best %s configuration found after %zd trials:", opname, trials); \
			LOG2(1, " Block (%-4u, %4u)", bestBlock.x, bestBlock.y); \
			LOG2(1, " Grid  (%-4u, %4u)", bestGrid.x, bestGrid.y); \
			LOG2(1, " Min time: %.4f ms", minRuntime); \
			fflush(stdout); \
		} \
	} while(0)

	void tune_kernel(
		void (*kernel)(
		const 	size_t, 
		const 	size_t, 
				Table*),
		const 	char* 	opname,
				dim3& 	bestBlock,
				dim3& 	bestGrid,
		const 	size_t& offset,
		const 	size_t& size,
				Table* 	ps)
	{
		size_t shared_element_bytes = 0;
		TUNE_1D(offset, size, ps);
	}

	void tune_step(
				dim3& 				bestBlock,
				dim3& 				bestGrid,
		const 	size_t& 			shared_element_bytes,
		const 	bool& 				shared_size_yextend,
		const 	size_t& 			data_size_in_x,
		const 	size_t& 			data_size_in_y,
				const_refs_t 		gate_refs,
				const_buckets_t 	gate_buckets,
				Tableau& 			tableau)
	{
		assert(gate_ref_t(data_size_in_x) == data_size_in_x);
		const char* opname = "step";
		size_t _initThreadsPerBlockX = initThreadsPerBlockX;
		initThreadsPerBlockX = 1;
		TUNE_2D_CALL(
			call_step_2D,
			gate_refs, 
			gate_buckets,
			tableau,
			data_size_in_x, 
			data_size_in_y);
		initThreadsPerBlockX = _initThreadsPerBlockX;
	}
	
	void tune_identity(
		void (*kernel)(
		const 	size_t, 
		const 	size_t, 
				Table*, 
				Table*),
				dim3& 	bestBlock,
				dim3& 	bestGrid,
		const 	size_t& offset,
		const 	size_t& size,
				Table* 	xs,
				Table* 	zs)
	{
		const char* opname = "identity";
		size_t shared_element_bytes = 0;
		TUNE_1D(offset, size, xs, zs);
	}
	
	void tune_reset_pivots(
		void (*kernel)(
				pivot_t*, 
		const 	size_t),
				dim3& 		bestBlock,
				dim3& 		bestGrid,
				pivot_t* 	pivots,
		const 	size_t 		size)
	{
		const char* opname = "reset_pivots";
		size_t shared_element_bytes = 0;
		TUNE_1D(pivots, size);
	}
	
	void tune_finding_all_pivots(
		void (*kernel)(
				pivot_t*,
				const_buckets_t,
				const_refs_t,
				const_table_t,
		const 	size_t,
		const 	size_t,
		const 	size_t,
		const 	size_t,
		const 	size_t),
				dim3& 				bestBlock,
				dim3& 				bestGrid,
		const 	size_t& 			shared_element_bytes,
		const 	bool& 				shared_size_yextend,
		const 	size_t& 			data_size_in_x,
		const 	size_t& 			data_size_in_y,
				pivot_t* 			pivots,
				const_buckets_t 	measurements,
				const_refs_t 		refs,
				const_table_t 		inv_xs,
		const 	size_t 				num_gates,
		const 	size_t 				num_qubits,
		const 	size_t 				num_words_major,
		const 	size_t 				num_words_minor,
		const 	size_t 				num_qubits_padded
	)
	{
		const char* opname = "finding all pivots";
		TUNE_2D(pivots, measurements, refs, inv_xs, num_gates, num_qubits, num_words_major, num_words_minor, num_qubits_padded);
	}
	
	void tune_finding_new_pivots(
		void (*kernel)(
				pivot_t*,
				const_table_t,
		const 	qubit_t,
		const 	size_t,
		const 	size_t,
		const 	size_t,
		const 	size_t),
				dim3& 				bestBlock,
				dim3& 				bestGrid,
		const 	size_t& 			shared_element_bytes,
				pivot_t* 			pivots,
				const_table_t 		inv_xs,
		const 	qubit_t& 			qubit,
		const 	size_t& 			size,
		const 	size_t 				num_words_major,
		const 	size_t 				num_words_minor,
		const 	size_t 				num_qubits_padded)
	{
		const char* opname = "finding new pivots";
		TUNE_1D(pivots, inv_xs, qubit, size, num_words_major, num_words_minor, num_qubits_padded);
	}

	void tune_inject_swap(
		void (*kernel)(
				Table*, 
				Table*,
				Signs*,
				const_pivots_t,
		const 	size_t, 
		const 	size_t, 
		const 	size_t),
				dim3& 			bestBlock,
				dim3& 			bestGrid,
				Table* 			xs,
				Table* 			zs,
				Signs* 			ss,
				const_pivots_t 	pivots,
		const 	size_t& 		num_words_major,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		num_qubits_padded)
	{
		const char* opname = "injecting swap";
		size_t shared_element_bytes = 0;
		const size_t size = num_words_minor;
		TUNE_1D(xs, zs, ss, pivots, num_words_major, num_words_minor, num_qubits_padded);	
	}

	void tune_outplace_transpose(
		void (*kernel)(
				Table*, 
				Table*, 
				const_table_t, 
				const_table_t, 
		const 	size_t, 
		const 	size_t, 
		const 	size_t),
		const 	char* 				opname, 
				dim3& 				bestBlock, 
				dim3& 				bestGrid,
		const 	size_t& 			shared_element_bytes, 
		const 	bool& 				shared_size_yextend,
		const 	size_t& 			data_size_in_x, 
		const 	size_t& 			data_size_in_y,
				Table* 				xs1, 
				Table* 				zs1,
        		const_table_t 		xs2, 
				const_table_t 		zs2,
        const 	size_t& 			num_words_major, 
		const 	size_t& 			num_words_minor, 
		const 	size_t& 			num_qubits_padded,
		const 	bool&	 			row_major) 
	{
		const int64 _initThreadsPerBlockX = initThreadsPerBlockX;
		const int64 _initThreadsPerBlockY = initThreadsPerBlockY;
		const int64 _maxThreadsPerBlockY = maxThreadsPerBlockY;
		const int64 _maxThreadsPerBlockX = maxThreadsPerBlockX;
		if (row_major) {
			initThreadsPerBlockX = 64;
			maxThreadsPerBlockY = 32;
		} else {
			initThreadsPerBlockY = 32;
			maxThreadsPerBlockX = 32;
		}
		TUNE_2D(xs1, zs1, xs2, zs2, num_words_major, num_words_minor, num_qubits_padded);
		initThreadsPerBlockX = _initThreadsPerBlockX;
		initThreadsPerBlockY = _initThreadsPerBlockY;
		maxThreadsPerBlockY = _maxThreadsPerBlockY;
		maxThreadsPerBlockX = _maxThreadsPerBlockX;
	}

	void tune_inplace_transpose(
		void (*transpose_tiles_kernel)(
				Table*, 
				Table*, 
		const 	size_t, 
		const 	size_t, 
		const 	bool),
		void (*swap_tiles_kernel)(
				Table*, 
				Table*, 
		const 	size_t, 
		const 	size_t),
				dim3& 		bestBlockTransposeBits, 
				dim3& 		bestGridTransposeBits,
				dim3& 		bestBlockTransposeSwap, 
				dim3& 		bestGridTransposeSwap,
				Table*		xs, 
				Table* 		zs,
        const 	size_t& 	num_words_major, 
		const 	size_t& 	num_words_minor, 
		const 	bool& 		row_major) 
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

	void tune_single_pass(
				dim3&       	bestBlock, 
				dim3&       	bestGrid,
		const   size_t&     	shared_element_bytes, 
		const   size_t&     	data_size_in_x, 
		const   size_t&     	data_size_in_y,
				SINGLE_PASS_ARGS,
		const   size_t&     	num_chunks,
		const   size_t&     	num_words_minor,
		const   size_t&     	max_blocks)
	{
		const char* opname = "scan single pass";
		size_t _initThreadsPerBlockY = initThreadsPerBlockY;
		initThreadsPerBlockY = 1;
		TUNE_2D_PREFIX_SINGLE_CUB(
			call_single_pass_kernel,
			SINGLE_PASS_INPUT,
			num_chunks, 
			num_words_minor,
			max_blocks
		);
		initThreadsPerBlockY = _initThreadsPerBlockY;
	}

	void tune_prefix_pass_1(
				dim3&       bestBlock, 
				dim3&       bestGrid,
		const   size_t&     shared_element_bytes, 
		const   size_t&     data_size_in_x, 
		const   size_t&     data_size_in_y,
				PASS_1_ARGS_PREFIX,
		const   size_t&     num_blocks,
		const   size_t&     num_words_minor,
		const   size_t&     max_blocks,
		const   size_t&     max_sub_blocks) 
	{
		const char* opname = "prefix pass 1";
		size_t _initThreadsPerBlockY = initThreadsPerBlockY;
		initThreadsPerBlockY = 1;
		TUNE_2D_PREFIX_CUB(
			false,
			call_scan_blocks_pass_1_kernel,
			MULTI_PASS_INPUT,
			num_blocks, 
			num_words_minor,
			max_blocks,
			max_sub_blocks);
		initThreadsPerBlockY = _initThreadsPerBlockY;
	}

	void tune_prefix_pass_2(
		void (*kernel)(
				PASS_2_ARGS_PREFIX,
		const 	size_t, 
		const 	size_t, 
		const 	size_t, 
		const 	size_t, 
		const 	size_t),
				dim3& 			bestBlock, 
				dim3& 			bestGrid,
		const 	size_t& 		data_size_in_x, 
		const 	size_t& 		data_size_in_y,
				PASS_2_ARGS_PREFIX,
		const 	size_t& 		num_blocks,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		max_blocks,
		const 	size_t& 		max_sub_blocks,
		const 	size_t& 		pass_1_blocksize) 
	{
		const char* opname = "prefix pass 2";
		const size_t shared_element_bytes = 0;
		const bool shared_size_yextend = false;
		TUNE_2D(
			MULTI_PASS_INPUT,
			num_blocks, 
			num_words_minor, 
			max_blocks, 
			max_sub_blocks,
			pass_1_blocksize);
	}

	void tune_inject_pass_1(
				dim3&           bestBlock, 
				dim3&           bestGrid,
		const   size_t&         shared_element_bytes, 
		const   size_t&         data_size_in_x, 
		const   size_t&         data_size_in_y,
				CALL_ARGS_GLOBAL_PREFIX,
				Tableau& 		input, 
		const   pivot_t*        pivots,
		const   size_t&         active_targets,
		const   size_t&         num_words_major,
		const   size_t&         num_words_minor,
		const   size_t&         num_qubits_padded,
		const   size_t&         max_blocks)
	{
		const char* opname = "inject-cx pass 1";
		size_t _initThreadsPerBlockY = initThreadsPerBlockY;
		initThreadsPerBlockY = 1;
		TUNE_2D_PREFIX_CUB(
			false,
			call_injectcx_pass_1_kernel,
			CALL_INPUT_GLOBAL_PREFIX,
			input,
			pivots,
			active_targets,
			num_words_major,
			num_words_minor,
			num_qubits_padded,
			max_blocks);
		initThreadsPerBlockY = _initThreadsPerBlockY;
	}

	void tune_inject_pass_2(
				dim3& 			bestBlock, 
				dim3& 			bestGrid,
		const 	size_t& 		shared_element_bytes, 
		const 	size_t& 		data_size_in_x, 
		const 	size_t& 		data_size_in_y,
				CALL_ARGS_GLOBAL_PREFIX,
				Tableau& 		input, 
		const 	pivot_t* 		pivots,
		const 	size_t& 		active_targets,
		const 	size_t& 		num_words_major,
		const 	size_t& 		num_words_minor,
		const 	size_t& 		num_qubits_padded,
		const 	size_t& 		max_blocks,
		const 	size_t& 		pass_1_blocksize) 
	{
		const char* opname = "inject-cx pass 2";
		const bool shared_size_yextend = true;
		TUNE_2D_CUB(
			call_injectcx_pass_2_kernel,
			CALL_INPUT_GLOBAL_PREFIX,
			input,
			pivots,
			active_targets,
			num_words_major,
			num_words_minor,
			num_qubits_padded,
			max_blocks,
			pass_1_blocksize
		);
	}

}

