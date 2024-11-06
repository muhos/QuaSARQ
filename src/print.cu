#include "simulator.hpp"
#include "print.cuh"

namespace QuaSARQ {

	NOINLINE_DEVICE void REPCH_GPU(const char* ch, const size_t& size, const size_t& off) {
        for (size_t i = off; i < size; i++) LOGGPU("%s", ch);
    }

        NOINLINE_ALL void print_table(const Table& t) {
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

    NOINLINE_ALL void print_table(const Table& t, const size_t& num_qubits, const size_t& num_words_major, const size_t& num_words_minor) {
        LOGGPU("%-3s ", "g\\q ");
        size_t bits, rows, cols;
        if (t.is_rowmajor()) {
            bits = num_words_minor * WORD_BITS;
            rows = 2 * num_qubits, cols = num_words_minor;
        }
        else {
            bits = num_words_major * WORD_BITS;
            rows = num_qubits, cols = num_words_major;
        }
        for (size_t q = 0; q < bits; q++) {
            if (q > 0 && q % WORD_BITS == 0)
                LOGGPU("  ");
            LOGGPU("%-3lld", q);
        }
        LOGGPU("\n\n");
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

    NOINLINE_ALL void print_table_signs(const Signs& ss, const size_t& offset) {
        if (ss.is_unpacked()) {
            LOGGPU("g\n");
            for (size_t i = offset; i < ss.size(); i++) {
                LOGGPU("%-3lld   %-2d\n", i, ss.unpacked_data()[i]);      
            }
            LOGGPU("\n");
        }
        else {
            LOGGPU("     ");
            for (size_t i = offset; i < ss.size(); i++) {
                #if defined(WORD_SIZE_64)
                LOGGPU(B2B_STR, RB2B(uint32(word_std_t(ss[i]) & 0xFFFFFFFFUL)));
                LOGGPU(B2B_STR, RB2B(uint32((word_std_t(ss[i]) >> 32) & 0xFFFFFFFFUL)));
                #else
                LOGGPU(B2B_STR, RB2B(word_std_t(ss[i])));
                #endif
                LOGGPU("  ");
            }
            LOGGPU("\n");
        }
    }

    NOINLINE_ALL void print_tables(const Table& xs, const Table& zs, const Signs& ss, const size_t& num_qubits, const int64& level, const bool& measuring) {
        if (measuring) 
            LOGGPU(" ---[ Destab/stab X-Table at (%-2lld)-step ]---------------------\n", level);
        else 
            LOGGPU(" ---[ X-Table at (%-2lld)-step ]---------------------\n", level);
        print_table(xs, num_qubits, xs.num_words_major(), xs.num_words_minor());
        if (measuring) 
            LOGGPU(" ---[ Destab/stab Z-Table at (%-2lld)-step ]---------------------\n", level);
        else 
            LOGGPU(" ---[ Z-Table at (%-2lld)-step ]---------------------\n", level);
        print_table(zs, num_qubits, zs.num_words_major(), zs.num_words_minor());
        LOGGPU(" ---[ Signs at (%-2lld)-step ]-----------------------\n", level);
        print_table_signs(ss);
    }

    NOINLINE_ALL void print_tables(const Table& ps, const Signs& ss, const size_t& num_qubits, const int64& level, const bool& measuring) {
        LOGGPU(" ---[ XZ bits at (%-2lld)-step ]---------------------\n", level);
        print_table(ps);
        LOGGPU(" ---[ Signs at (%-2lld)-step   ]---------------------\n", level);
        print_table_signs(ss);
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

    NOINLINE_DEVICE void print_shared_aux(DeviceLocker& dlocker, const Gate& m, byte_t* smem, const size_t& copied_row, const size_t& multiplied_row) {
        dlocker.lock();
        const grid_t tx = threadIdx.x, BX = blockDim.x;
        const grid_t shared_tid = threadIdx.y * BX * 2 + tx;
        word_std_t* aux = reinterpret_cast<word_std_t*>(smem);
        int* aux_power = reinterpret_cast<int*>(aux + blockDim.y * blockDim.x * 2);
        word_std_t* aux_xs = aux;
        word_std_t* aux_zs = aux + blockDim.x;
        int* pos_is = aux_power;
        int* neg_is = aux_power + blockDim.x;
        if (!global_tx) {
            if (multiplied_row == UINT64_MAX)
                LOGGPU("qubit(%d), copied row(%lld):\n", m.wires[0], copied_row);
            else 
                LOGGPU("qubit(%d), row(%lld) x row(%lld):\n", m.wires[0], copied_row, multiplied_row);
        }
        LOGGPU(" shared_tid(%lld): aux_xs[b:%lld, t:%lld]: " B2B_STR "\n", shared_tid, BX, tx, RB2B(aux_xs[shared_tid]));
        LOGGPU(" shared_tid(%lld): aux_zs[b:%lld, t:%lld]: " B2B_STR "\n", shared_tid, BX, tx, RB2B(aux_zs[shared_tid]));
        LOGGPU(" shared_tid(%lld): pos_i[b:%lld, t:%lld] = %d\n", shared_tid, BX, tx, pos_is[shared_tid]);
        LOGGPU(" shared_tid(%lld): neg_i[b:%lld, t:%lld] = %d\n", shared_tid, BX, tx, neg_is[shared_tid]);
        dlocker.unlock();
    }

	__global__ void print_tableau_k(const Table* ps, const Signs* ss, const size_t num_qubits, const depth_t level, const bool measuring) {
		if (!global_tx) {
			print_tables(*ps, *ss, num_qubits, level == MAX_DEPTH ? -1 : int64(level), measuring);
		}
	}

	__global__ void print_tableau_k(const Table* xs, const Table* zs, const Signs* ss, const size_t num_qubits, const depth_t level, const bool measuring) {
		if (!global_tx) {
			print_tables(*xs, *zs, *ss, num_qubits, level == MAX_DEPTH ? -1 : int64(level), measuring);
		}
	}

	__global__ void print_paulis_k(const Table* xs, const Table* zs, const Signs* ss, const size_t num_words_major, const size_t num_qubits, const bool extended) {
		if (!global_tx) {
			print_state(*xs, *zs, *ss, 0, num_qubits, num_qubits, num_words_major);
			if (extended) {
				REPCH_GPU("-", num_qubits + 1);
				LOGGPU("\n");
				print_state(*xs, *zs, *ss, num_qubits, 2*num_qubits, num_qubits, num_words_major);
			}
		}
	}
	
	__global__ void print_paulis_k(const Table* ps, const Signs* ss, const size_t num_words_major, const size_t num_qubits, const depth_t level) {
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


	__global__ void print_gates_k(const gate_ref_t* refs, const bucket_t* gates, const Pivot* pivots, const gate_ref_t num_gates) {
		if (!global_tx) {
			for (gate_ref_t i = 0; i < num_gates; i++) {
				const gate_ref_t r = refs[i];
				LOGGPU("  Gate(%3d , r:%3d):", i, r);
				const Gate &gate = (Gate &)gates[r];
				gate.print();
                if (gate.type == M) {
                    REPCH_GPU(" ", 25);
                    LOGGPU("pivot");
                    pivots[i].print();
                }
			}
		}
	}

	__global__ void print_measurements_k(const gate_ref_t* refs, const bucket_t* measurements, const Pivot* pivots, const gate_ref_t num_gates) {
		if (!global_tx) {
			for (gate_ref_t i = 0; i < num_gates; i++) {
				const gate_ref_t r = refs[i];
				const Gate &m = (Gate &)measurements[r];
				LOGGPU(" %8d     %10s    %2c\n", m.wires[0], 
					pivots[i].indeterminate == INVALID_PIVOT ? "definite" : "random",  
					m.measurement != UNMEASURED ? char(((m.measurement % 4 + 4) % 4 >> 1) + 48) : 'U');
				// LOGGPU(" %8d     %10s    %2d\n", m.wires[0], 
				// 	m.pivot == MAX_QUBITS ? "definite" : "random",  
				// 	m.measurement != UNMEASURED ? m.measurement : -1);
			}
		}
	}

	void Simulator::print_paulis(const Tableau<DeviceAllocator>& tab, const depth_t& depth_level, const bool& reversed) {
		if (!options.sync) SYNCALL;
		if (depth_level == -1) 
			LOGHEADER(0, 3, "Initial state");
		else if (options.print_step_state)
			LOG2(0, "State after %d-step", depth_level);
		else if (options.print_final_state)
			LOGHEADER(0, 3, "Final state");
		if (num_qubits > 100) {
            LOGWARNING("State is too large to print.");
			fflush(stdout);
		}
        print_paulis_k << <1, 1 >> > (XZ_TABLE(tab), tab.signs(), tab.num_words_major(), num_qubits, measuring);
        LASTERR("failed to launch print-paulis kernel");
        SYNCALL;
        fflush(stdout);
	}

	void Simulator::print_tableau(const Tableau<DeviceAllocator>& tab, const depth_t& depth_level, const bool& reversed) {
		if (!options.sync) SYNCALL;
		LOG2(0, "");
		if (depth_level == -1)
			LOG2(0, "Initial tableau before simulation");
		else if (depth_level == depth)
			LOG2(0, "Final tableau after %d %ssimulation steps", depth, reversed ? "reversed " : "");
		else
			LOG2(0, "Tableau after %d-step", depth_level);
        print_tableau_k << <1, 1 >> > (XZ_TABLE(tab), tab.signs(), num_qubits, depth_level, measuring);
        LASTERR("failed to launch print-tableau kernel");
        SYNCALL;
        fflush(stdout);
	}

	void Simulator::print_gates(const DeviceCircuit<DeviceAllocator>& gpu_circuit, const gate_ref_t& num_gates, const depth_t& depth_level) {
		if (!options.print_gates) return;
		if (!options.sync) SYNCALL;
		LOG2(0, " Gates on GPU for %d-time step:", depth_level);
		print_gates_k << <1, 1 >> > (gpu_circuit.references(), gpu_circuit.gates(), gpu_circuit.pivots(), num_gates);
		LASTERR("failed to launch print-gates kernel");
		SYNCALL;
		fflush(stdout);
	}

	void Simulator::print_measurements(const DeviceCircuit<DeviceAllocator>& gpu_circuit, const gate_ref_t& num_gates, const depth_t& depth_level) {
		if (!options.print_measurements) return;
		if (!circuit.is_measuring(depth_level)) return;
		if (!options.sync) SYNCALL;
		LOG2(0, " Measurements on GPU for %d-time step:", depth_level);
		LOG2(0, "%10s   %10s     %5s", "Qubit", "Type", "Outcome");
		print_measurements_k << <1, 1 >> > (gpu_circuit.references(), gpu_circuit.gates(), gpu_circuit.pivots(), num_gates);
		LASTERR("failed to launch print-measurements kernel");
		SYNCALL;
		fflush(stdout);
	}

}

