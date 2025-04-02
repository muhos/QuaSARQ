#include "simulator.hpp"
#include "print.cuh"
#include "access.cuh"

namespace QuaSARQ {

	NOINLINE_DEVICE void REPCH_GPU(const char* ch, const size_t& size, const size_t& off) {
        for (size_t i = off; i < size; i++) LOGGPU("%s", ch);
    }

        NOINLINE_ALL void print_table_interleave(const Table& t) {
        const bool is_rowmajor = t.is_rowmajor();
        const size_t size = t.size();
        const size_t major_end = t.num_words_major();
        const size_t minor_end = t.num_words_major() * WORD_BITS;
        #if defined(INTERLEAVE_XZ)
        const size_t interleaving_offset = major_end * INTERLEAVE_COLS * 2;
        #else
        const size_t interleaving_offset = major_end;
        #endif
        size_t j = 0;
        for (size_t i = 0; i < size; i++) {
            if (i > 0 && i % major_end == 0)
                LOGGPU("\n");
            if (i > 0 && i % minor_end == 0)
                LOGGPU("\n");
            LOGGPU("  ");
            
            if (i > 0 && i % interleaving_offset == 0) 
                j++;
            if (i % major_end == 0)
                LOGGPU("%-2lld ", j);
            #if defined(WORD_SIZE_64)
            LOGGPU(B2B_STR, RB2B(uint32(word_std_t(t[i]) & 0xFFFFFFFFUL)));
            LOGGPU(B2B_STR, RB2B(uint32((word_std_t(t[i]) >> 32) & 0xFFFFFFFFUL)));
            #else
            LOGGPU(B2B_STR, RB2B(word_std_t(t[i])));
            #endif
        }
        LOGGPU("\n");
    }

    NOINLINE_ALL void print_table(const Table& t, const size_t& total_targets) {
        LOGGPU("%-3s ", "g\\q ");
        size_t bits, rows, cols, word_idx;
        const size_t num_qubits_padded = t.num_qubits_padded();
        const size_t num_words_major = t.num_words_major();
        const size_t num_words_minor = t.num_words_minor();
        constexpr int ROWMAJOR_STEP = 2;
        if (t.is_rowmajor()) {
            bits = num_words_minor * WORD_BITS;
            rows = total_targets ? total_targets : 2 * num_qubits_padded, cols = num_words_minor;
        }
        else {
            bits = num_words_major * WORD_BITS;
            rows = num_qubits_padded, cols = num_words_major;
        }
        for (size_t q = 0; q < bits; q++) {
            if (q > 0 && q % WORD_BITS == 0)
                LOGGPU("  ");
            LOGGPU("%-3lld", q);
        }
        LOGGPU("\n\n");
        if (t.is_rowmajor()) {
            for (size_t q = 0; q < rows; q++) {
                LOGGPU("%-3lld  ", q);
                for (size_t w = 0; w < cols; w++) {
                    const size_t word_idx = q + w * rows;
                    #if defined(WORD_SIZE_64)
                    LOGGPU(B2B_STR, RB2B(uint32(word_std_t(t[word_idx]) & 0xFFFFFFFFUL)));
                    LOGGPU(B2B_STR "  ", RB2B(uint32((word_std_t(t[word_idx]) >> 32) & 0xFFFFFFFFUL)));
                    #else
                    LOGGPU(B2B_STR "  ", RB2B(word_std_t(t[word_idx])));
                    #endif
                }
                LOGGPU("\n");
            }
        }
        else {
            for (size_t q = 0; q < rows; q++) {
                LOGGPU("%-3lld  ", q);
                for (size_t w = 0; w < cols; w++) {
                    const size_t word_idx = q * cols + w;
                    #if defined(WORD_SIZE_64)
                    LOGGPU(B2B_STR, RB2B(uint32(word_std_t(t[word_idx]) & 0xFFFFFFFFUL)));
                    LOGGPU(B2B_STR "  ", RB2B(uint32((word_std_t(t[word_idx]) >> 32) & 0xFFFFFFFFUL)));
                    #else
                    LOGGPU(B2B_STR "  ", RB2B(word_std_t(t[word_idx])));
                    #endif
                }
                LOGGPU("\n");
            }
        }
        
    }

    NOINLINE_ALL void print_table_signs(const Signs& ss, const size_t& start, const size_t& end) {
        const size_t size = ss.is_unpacked() ? ss.size() : ss.size() * WORD_BITS;
        for (size_t i = start; i < end; i++) {
            LOGGPU("g%-3lld   %-2d\n", i >= ss.num_qubits_padded() ? (i - ss.num_qubits_padded()) : i,  ss.is_unpacked() ? ss.unpacked_data()[i] : bool(ss[WORD_OFFSET(i)] & sign_t(BITMASK_GLOBAL(i))));
        }
    }

    NOINLINE_ALL void print_tables(const Table& xs, const Table& zs, const Signs* ss, const int64& level) {
        LOGGPU(" ---[ %s X-Table at (%-2lld)-step ]---------------------\n", xs.is_rowmajor() ? "Transposed" : "", level);
        print_table(xs);
        LOGGPU(" ---[ %s Z-Table at (%-2lld)-step ]---------------------\n", zs.is_rowmajor() ? "Transposed" : "", level);
        print_table(zs);
        if (ss != nullptr) {
            LOGGPU(" ---[ Signs at (%-2lld)-step ]-----------------------\n", level);
            const size_t unfolded_size = ss->is_unpacked() ? ss->size() : ss->size() * WORD_BITS;
            if (ss->num_qubits_padded() == unfolded_size)
                print_table_signs(*ss, 0, ss->num_qubits_padded());
            else {
                assert(2 * ss->num_qubits_padded() == unfolded_size);
                LOGGPU("Destabilizers:\n");
                print_table_signs(*ss, 0, ss->num_qubits_padded());
                LOGGPU("Stabilizers:\n");
                print_table_signs(*ss, ss->num_qubits_padded(), 2 * ss->num_qubits_padded());
            }
        }
        LOGGPU("\n");
    }

    NOINLINE_ALL void print_state(const Table& xs, const Table& zs, const Signs& ss, 
                                const size_t& start, const size_t& end, 
                                const size_t& num_qubits, const size_t& num_words_major) {
        for (size_t w = start; w < end; w++) {
            const word_t pow2 = BITMASK_GLOBAL(w);
            if (ss[WORD_OFFSET(w)] & sign_t(pow2)) {
                LOGGPU("-");
            }
            else {
                LOGGPU("+");
            }
            for (size_t q = 0; q < num_qubits; q++) {
                const size_t word_idx = q * num_words_major + WORD_OFFSET(w);
                if ((!(xs[word_idx] & pow2)) && (!(zs[word_idx] & pow2)))
                    LOGGPU("I");
                if ((xs[word_idx] & pow2) && (!(zs[word_idx] & pow2)))
                    LOGGPU("X");
                if ((!(xs[word_idx] & pow2)) && (zs[word_idx] & pow2))
                    LOGGPU("Z");
                if ((xs[word_idx] & pow2) && (zs[word_idx] & pow2))
                    LOGGPU("Y");
            }
            LOGGPU("\n");
        }
    }

    NOINLINE_DEVICE void print_column(DeviceLocker& dlocker, const Table& xs, const Table& zs, const Signs& ss, const size_t& q, const size_t& num_qubits, const size_t& num_words_major) {
        dlocker.lock();
        LOGGPU("   X(%-2lld)   Z(%-2lld)   S\n", q, q);
        for (size_t i = 0; i < 2 * num_qubits; i++) {
            if (i == num_qubits) {
                REPCH_GPU("-", 20);
                LOGGPU("\n");
            }
            LOGGPU("%-2lld   %-2d     %-2d     %-2d\n", i,
                bool(word_std_t(xs[q * num_words_major + WORD_OFFSET(i)]) & BITMASK_GLOBAL(i)),
                bool(word_std_t(zs[q * num_words_major + WORD_OFFSET(i)]) & BITMASK_GLOBAL(i)),
                bool(word_std_t(ss[WORD_OFFSET(i)]) & BITMASK_GLOBAL(i)));
        }
        dlocker.unlock();
    }

    NOINLINE_DEVICE void print_row(DeviceLocker& dlocker, const Gate& m, const Table& inv_xs, const Table& inv_zs, const int* inv_ss, const size_t& gen_idx, const size_t& num_words_minor) {
        dlocker.lock();
        m.print();
        LOGGPU(" X(%lld): ", gen_idx);
        const size_t row = gen_idx * num_words_minor;
        for (size_t i = 0; i < num_words_minor; i++) {
            #if defined(WORD_SIZE_64)
            LOGGPU(B2B_STR, RB2B(uint32(word_std_t(inv_xs[row + i]) & 0xFFFFFFFFUL)));
            LOGGPU(B2B_STR, RB2B(uint32((word_std_t(inv_xs[row + i]) >> 32) & 0xFFFFFFFFUL)));
            #else
            LOGGPU(B2B_STR, RB2B(word_std_t(inv_xs[row + i])));
            #endif
            LOGGPU("  ");
        }
        LOGGPU("\n Z(%lld): ", gen_idx);
        for (size_t i = 0; i < num_words_minor; i++) {
            #if defined(WORD_SIZE_64)
            LOGGPU(B2B_STR, RB2B(uint32(word_std_t(inv_zs[row + i]) & 0xFFFFFFFFUL)));
            LOGGPU(B2B_STR, RB2B(uint32((word_std_t(inv_zs[row + i]) >> 32) & 0xFFFFFFFFUL)));
            #else
            LOGGPU(B2B_STR, RB2B(word_std_t(inv_zs[row + i])));
            #endif
            LOGGPU("  ");
        }
        LOGGPU("\n S(%lld): %d\n", gen_idx, inv_ss[gen_idx]);
        dlocker.unlock();
    }

	__global__ void print_tableau_k(ConstTablePointer xs, ConstTablePointer zs, ConstSignsPointer ss, const depth_t level) {
		if (!global_tx) {
			print_tables(*xs, *zs, ss, level == MAX_DEPTH ? -1 : int64(level));
		}
	}

	__global__ void print_paulis_k(ConstTablePointer xs, ConstTablePointer zs, ConstSignsPointer ss, const size_t num_words_major, const size_t num_qubits, const bool extended) {
		if (!global_tx) {
			print_state(*xs, *zs, *ss, 0, num_qubits, num_qubits, num_words_major);
			if (extended) {
				REPCH_GPU("-", num_qubits + 1);
				LOGGPU("\n");
				print_state(*xs, *zs, *ss, num_qubits, 2*num_qubits, num_qubits, num_words_major);
			}
		}
	}
	
	__global__ void print_paulis_k(ConstTablePointer ps, ConstSignsPointer ss, const size_t num_words_major, const size_t num_qubits, const depth_t level) {
		if (!global_tx) {
			const word_t *words = ps->data();
			for (size_t w = 0; w < num_qubits; w++) {
				const word_t pow2 = BITMASK_GLOBAL(w);
				for (size_t q = 0; q < num_qubits; q++) {
					if (q == 0 && (*ss)[WORD_OFFSET(q)] & sign_t(pow2)) {
						LOGGPU("-");
					}
					else if (q == 0) {
						LOGGPU("+");
					}
					const size_t x_word_idx = X_OFFSET(q) * num_words_major + X_WORD_OFFSET(WORD_OFFSET(w));
					const size_t z_word_idx = Z_OFFSET(q) * num_words_major + Z_WORD_OFFSET(WORD_OFFSET(w));
					if ((!(words[x_word_idx] & pow2)) && (!(words[z_word_idx] & pow2)))
						LOGGPU("I");
					if ((words[x_word_idx] & pow2) && (!(words[z_word_idx] & pow2)))
						LOGGPU("X");
					if ((words[x_word_idx] & pow2) && (words[z_word_idx] & pow2))
						LOGGPU("Y");
					if ((!(words[x_word_idx] & pow2)) && (words[z_word_idx] & pow2))
						LOGGPU("Z");
				}
				LOGGPU("\n");
			}
		}
	}


	__global__ void print_gates_k(ConstRefsPointer refs, ConstBucketsPointer gates, ConstPivotsPointer pivots, const gate_ref_t num_gates) {
		if (!global_tx) {
			for (gate_ref_t i = 0; i < num_gates; i++) {
				const gate_ref_t r = refs[i];
				LOGGPU("  Gate(%3d , r:%3d):", i, r);
				const Gate &gate = (Gate &)gates[r];
				gate.print();
                if (gate.type == M) {
                    REPCH_GPU(" ", 25);
                    LOGGPU("pivot: %d", pivots[i]);
                }
			}
		}
	}

	__global__ void print_measurements_k(ConstRefsPointer refs, ConstBucketsPointer measurements, ConstPivotsPointer pivots, const gate_ref_t num_gates) {
        for_parallel_x(i, num_gates) {
            const gate_ref_t r = refs[i];
            const Gate &m = (Gate &)measurements[r];
            LOGGPU(" q%-10d: %c (%s)\n", m.wires[0], 
                m.measurement != UNMEASURED ? char(((m.measurement % 4 + 4) % 4 >> 1) + 48) : 'U',
                pivots[i] == INVALID_PIVOT ? "definite" : "random");
            // LOGGPU(" %8d     %10s    %2d\n", m.wires[0], 
            // 	m.pivot == MAX_QUBITS ? "definite" : "random",  
            // 	m.measurement != UNMEASURED ? m.measurement : -1);
        }
	}

	void Simulator::print_paulis(const Tableau& tab, const depth_t& depth_level, const bool& reversed) {
		if (!options.sync) SYNCALL;
		if (depth_level == -1) 
			LOGHEADER(0, 3, "Initial state");
		else if (options.print_stepstate)
			LOG2(0, "State after %d-step", depth_level);
		else if (options.print_finalstate)
			LOGHEADER(0, 3, "Final state");
		if (num_qubits > 100) {
            LOGWARNING("State is too large to print.");
			fflush(stdout);
		}
        print_paulis_k << <1, 1 >> > (XZ_TABLE(tab), tab.signs(), tab.num_words_major(), num_qubits, measuring);
        LASTERR("failed to launch print_paulis_k kernel");
        SYNCALL;
        fflush(stdout);
	}

	void Simulator::print_tableau(const Tableau& tab, const depth_t& depth_level, const bool& reversed, const bool& prefix) {
		if (!options.sync) SYNCALL;
		LOG2(0, "");
		if (depth_level == -1)
			LOG2(0, "Initial tableau before simulation");
		else if (depth_level == depth)
			LOG2(0, "Final tableau after %d %ssimulation steps", depth, reversed ? "reversed " : "");
		else
			LOG2(0, "Tableau after %d-step", depth_level);
        print_tableau_k << <1, 1 >> > (XZ_TABLE(tab), prefix ? nullptr : tab.signs(), depth_level);
        LASTERR("failed to launch print_tableau_k kernel");
        SYNCALL;
        fflush(stdout);
	}

	void Simulator::print_gates(const DeviceCircuit& gpu_circuit, const gate_ref_t& num_gates, const depth_t& depth_level) {
		if (!options.print_gates) return;
		if (!options.sync) SYNCALL;
		LOG2(0, " Gates on GPU for %d-time step:", depth_level);
		print_gates_k << <1, 1 >> > (gpu_circuit.references(), gpu_circuit.gates(), gpu_circuit.pivots(), num_gates);
		LASTERR("failed to launch print_gates_k kernel");
		SYNCALL;
		fflush(stdout);
	}

	void Simulator::print_measurements(const DeviceCircuit& gpu_circuit, const gate_ref_t& num_gates, const depth_t& depth_level) {
		if (!options.print_measurements) return;
		if (!circuit.is_measuring(depth_level)) return;
		if (!options.sync) SYNCALL;
		if (!options.progress_en) LOG2(0, " Measurements on GPU for %d-time step:", depth_level);
        else SETCOLOR(CLBLUE, stdout);
        uint32 currentblock = 256, currentgrid;
        OPTIMIZEBLOCKS(currentgrid, num_gates, currentblock);
		print_measurements_k <<< currentgrid, currentblock >>> (gpu_circuit.references(), gpu_circuit.gates(), gpu_circuit.pivots(), num_gates);
		LASTERR("failed to launch print_measurements_k kernel");
		SYNCALL;
        SETCOLOR(CNORMAL, stdout);
		fflush(stdout);
	}

}

