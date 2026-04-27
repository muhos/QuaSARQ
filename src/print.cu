#include "simulator.hpp"
#include "print.cuh"
#include "access.cuh"
#include "pivot.cuh"

namespace QuaSARQ {

    #define PRINT_HEX 0

	NOINLINE_ALL 
    void REPCH_GPU(
        const char*     ch, 
        const size_t&   size, 
        const size_t&   off) 
    {
        for (size_t i = off; i < size; i++) LOGGPU("%s", ch);
    }

    NOINLINE_ALL 
    void print_table(
        const Table&    t, 
        const size_t&   total_targets) 
    {
        const size_t num_qubits_padded = t.num_qubits_padded();
        const size_t num_words_major = t.num_words_major();
        const size_t num_words_minor = t.num_words_minor();
        #if ROW_MAJOR
        size_t bits, rows, cols;
        if (t.is_rowmajor()) {
            LOGGPU("%-3s ", "g\\q ");
            #if PRINT_HEX
            bits = num_words_minor;
            #else
            bits = num_words_minor * WORD_BITS;
            #endif
            rows = total_targets ? total_targets : 2 * num_qubits_padded, cols = num_words_minor;
        }
        else {
            LOGGPU("%-3s ", "q\\g ");
            #if PRINT_HEX
            bits = num_words_major;
            #else
            bits = num_words_major * WORD_BITS;
            #endif
            rows = num_qubits_padded, cols = num_words_major;
        }
        for (size_t q = 0; q < bits; q++) {
            #if PRINT_HEX
            LOGGPU("%-16lld   ", int64(q));
            #else
            if (q > 0 && q % WORD_BITS == 0)
                LOGGPU("  ");
            LOGGPU("%-3lld", int64(q));
            #endif
            if (q > 64) break;
        }
        LOGGPU("\n\n");
        if (t.is_rowmajor()) {
            for (size_t q = 0; q < rows; q++) {
                LOGGPU("%-3lld  ", int64(q));
                for (size_t w = 0; w < cols; w++) {
                    const size_t word_idx = q + w * rows;
                    #if PRINT_HEX
                    LOGGPU("0x%016llX ", uint64(t[word_idx]));
                    #else 
                    #if defined(WORD_SIZE_64)
                    LOGGPU(B2B_STR, RB2B(uint32(word_std_t(t[word_idx]) & 0xFFFFFFFFUL)));
                    LOGGPU(B2B_STR "  ", RB2B(uint32((word_std_t(t[word_idx]) >> 32) & 0xFFFFFFFFUL)));
                    #else
                    LOGGPU(B2B_STR "  ", RB2B(word_std_t(t[word_idx])));
                    #endif
                    #endif
                }
                LOGGPU("\n");
            }
        }
        else {
            for (size_t q = 0; q < rows; q++) {
                LOGGPU("%-3lld  ", int64(q));
                for (size_t w = 0; w < cols; w++) {
                    const size_t word_idx = q * cols + w;
                    #if PRINT_HEX
                    LOGGPU("0x%016llX ", uint64(t[word_idx]));
                    #else 
                    #if defined(WORD_SIZE_64)
                    LOGGPU(B2B_STR, RB2B(uint32(word_std_t(t[word_idx]) & 0xFFFFFFFFUL)));
                    LOGGPU(B2B_STR "  ", RB2B(uint32((word_std_t(t[word_idx]) >> 32) & 0xFFFFFFFFUL)));
                    #else
                    LOGGPU(B2B_STR "  ", RB2B(word_std_t(t[word_idx])));
                    #endif
                    #endif
                }
                LOGGPU("\n");
            }
        }
        #else
        size_t offset = 0;
        if (num_words_major == 2 * num_words_minor) {
            offset = num_words_minor;
            LOGGPU("Destabilizers:\n");
            for (size_t q = 0; q < num_qubits_padded; q++) {
                LOGGPU("%-3lld  ", int64(q));
                for (size_t w = 0; w < num_words_minor; w++) {
                    const size_t word_idx = q * num_words_major + w;
                    #if PRINT_HEX
                    LOGGPU("0x%016llX ", uint64(t[word_idx]));
                    #else 
                    #if defined(WORD_SIZE_64)
                    LOGGPU(B2B_STR, RB2B(uint32(word_std_t(t[word_idx]) & 0xFFFFFFFFUL)));
                    LOGGPU(B2B_STR "  ", RB2B(uint32((word_std_t(t[word_idx]) >> 32) & 0xFFFFFFFFUL)));
                    #else
                    LOGGPU(B2B_STR "  ", RB2B(word_std_t(t[word_idx])));
                    #endif
                    #endif
                }
                LOGGPU("\n");
            }
        }
        LOGGPU("Stabilizers:\n");
        const size_t cols = num_words_major < num_words_minor ? num_words_minor : num_words_major;
        for (size_t q = 0; q < num_qubits_padded; q++) {
            LOGGPU("%-3lld  ", int64(q));
            for (size_t w = 0; w < num_words_minor; w++) {
                const size_t word_idx = q * cols + w + offset;
                #if PRINT_HEX
                LOGGPU("0x%016llX ", uint64(t[word_idx]));
                #else 
                #if defined(WORD_SIZE_64)
                LOGGPU(B2B_STR, RB2B(uint32(word_std_t(t[word_idx]) & 0xFFFFFFFFUL)));
                LOGGPU(B2B_STR "  ", RB2B(uint32((word_std_t(t[word_idx]) >> 32) & 0xFFFFFFFFUL)));
                #else
                LOGGPU(B2B_STR "  ", RB2B(word_std_t(t[word_idx])));
                #endif
                #endif
            }
            LOGGPU("\n");
        }
        #endif
    }

    NOINLINE_ALL 
    void print_table_signs(
        const Signs&    ss, 
        const size_t&   start, 
        const size_t&   end) 
    {
        const size_t size = ss.is_unpacked() ? ss.size() : ss.size() * WORD_BITS;
        for (size_t i = start; i < end; i++) {
            LOGGPU("g%-3lld   %-2d\n", 
                (int64) (i >= ss.num_qubits_padded() ? i - ss.num_qubits_padded() : i),  
                ss.is_unpacked() ? ss.unpacked_data()[i] : bool(ss[WORD_OFFSET(i)] & sign_t(BITMASK_GLOBAL(i))));
        }
    }

    NOINLINE_ALL 
    void print_tables(
        const Table& xs, 
        const Table& zs, 
        const Signs* ss, 
        const int64& level) 
    {
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

    INLINE_ALL void __logchar(const char* s, FILE* f = nullptr) {
        if (f == nullptr)
            LOGGPU("%s", s);
        else {
            while (*s) fputc(*s++, f);
        }
    }

    #define LOGCHAR(...)  __logchar(__VA_ARGS__)

    NOINLINE_ALL
    void print_state(
        const Table&  xs,
        const Table&  zs,
        const Signs&  ss,
        size_t        num_qubits,
        size_t        num_words_major,
        size_t        offset,
        FILE*         f = nullptr
    )
    {
        for (size_t w = 0; w < num_qubits; ++w) {
            const word_t pow2 = BITMASK_GLOBAL(w);
            if (ss[offset + WORD_OFFSET(w)] & sign_t(pow2)) 
                LOGCHAR("-", f);
            else 
                LOGCHAR("+", f);

            for (size_t q = 0; q < num_qubits; ++q) {
                size_t idx = q * num_words_major + WORD_OFFSET(w) + offset;
                if (!(xs[idx] & pow2) && !(zs[idx] & pow2)) LOGCHAR("I", f);
                if  ( xs[idx] & pow2  && !(zs[idx] & pow2)) LOGCHAR("X", f);
                if (!(xs[idx] & pow2) &&  (zs[idx] & pow2)) LOGCHAR("Z", f);
                if  ( xs[idx] & pow2  &&  (zs[idx] & pow2)) LOGCHAR("Y", f);
            }
            LOGCHAR("\n", f);
        }
    }

	__global__ 
    void print_tableau_k(
        const_table_t   xs, 
        const_table_t   zs, 
        const_signs_t   ss, 
        const depth_t   level) 
    {
		if (!global_tx) {
			print_tables(*xs, *zs, ss, level == MAX_DEPTH ? -1 : int64(level));
		}
	}

	__global__ 
    void print_paulis_k(
        const_table_t   xs, 
        const_table_t   zs, 
        const_signs_t   ss, 
        const size_t    num_words_major, 
        const size_t    num_words_minor, 
        const size_t    num_qubits, 
        const bool      extended) 
    {
		if (!global_tx) {
			print_state(*xs, *zs, *ss, num_qubits, num_words_major, 0);
			if (extended) {
				REPCH_GPU("-", num_qubits + 1);
				LOGGPU("\n");
				print_state(*xs, *zs, *ss, num_qubits, num_words_major, num_words_minor);
			}
		}
	}

	__global__ 
    void print_gates_k(
        const_refs_t        refs, 
        const_buckets_t     gates, 
        const gate_ref_t    num_gates) 
    {
		if (!global_tx) {
			for (gate_ref_t i = 0; i < num_gates; i++) {
				const gate_ref_t r = refs[i];
				LOGGPU("  Gate(%3d , r:%3d):", i, r);
				const Gate &gate = (Gate &)gates[r];
				gate.print();
			}
		}
	}

    __global__ 
    void print_signs_k(
        const Signs* ss, 
        const int64& level) 
    {
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
        LOGGPU("\n");
    }

    inline string extract_circuit_name(const string& path) {
        size_t start = 0, end = path.size();
        for (size_t i = 0; i < path.size(); ++i) {
            char c = path[i];
            if (c == '/' || c == '\\')
                start = i + 1;
            else if (c == '.' && i >= start)
                end = i;
        }
        string name;
        if (end > start) {
            name.reserve(end - start);
            for (size_t i = start; i < end; ++i)
                name.push_back(path[i]);
        }
        return name;
    }

    void expand_statepath(string& path, const string& circuit) {
        const string placeholder = "<circuit>";
        const size_t pos =  path.find(placeholder);
        string expanded_path = path;
        if (pos != string::npos) {
            path.replace(pos, placeholder.size(), circuit);
        }
    }

	void Simulator::print_paulis(const Tableau& tab, const depth_t& depth_level, const bool& reversed) {
		if (!options.sync) SYNCALL;
		if (depth_level == MAX_DEPTH) 
			LOGHEADER(0, 3, "Initial state");
		else if (options.print_stepstate)
			LOG2(0, "State after %d-step", depth_level);
		else if (options.print_finalstate)
			LOGHEADER(0, 3, "Final state");
		if (num_qubits > options.print_limit) {
            LOGWARNING("State is too large to print on screen.\n"
                "The full state will be printed in options.statepath.\n"
                "Use --state-path to change file path.");
            string statepath = options.statepath;
            auto circuit_name = extract_circuit_name(circuit_path);
            expand_statepath(statepath, circuit_name);
            std::memcpy(options.statepath, statepath.c_str(), statepath.length());
            options.statepath[statepath.length()] = '\0';
            if (!open_file(state_file, options.statepath, "wb")) {
                LOGERROR("cannot open state file for writing.");
            }
            Table h_xs, h_zs; Signs h_ss;
            tab.copy_to_host(&h_xs, &h_zs, &h_ss);
            print_state(h_xs, h_zs, h_ss, num_qubits, tab.num_words_major(), 0, state_file);
			if (measuring) {
				REPCH('-', num_qubits + 1, state_file);
				LOGCHAR("\n", state_file);
				print_state(h_xs, h_zs, h_ss, num_qubits, tab.num_words_major(), tab.num_words_minor(), state_file);
			}
            close_file(state_file);
		}
        else {
            print_paulis_k << <1, 1 >> > (
                XZ_TABLE(tab), 
                tab.signs(), 
                tab.num_words_major(), 
                tab.num_words_minor(), 
                num_qubits, 
                measuring);
            LASTERR("failed to launch print_paulis_k kernel");
            SYNCALL;
            fflush(stdout);
        }
	}

	void Simulator::print_tableau(const Tableau& tab, const depth_t& depth_level, const bool& reversed, const bool& prefix) {
		if (!options.sync) SYNCALL;
		LOG2(0, "");
		if (depth_level == MAX_DEPTH)
			LOG2(0, "Initial tableau before simulation");
		else if (depth_level == depth)
			LOG2(0, "Final tableau after %d %ssimulation steps", depth, reversed ? "reversed " : "");
		else
			LOG2(0, "Tableau after %d-step", depth_level);
        print_tableau_k << <1, 1 >> > (
            XZ_TABLE(tab), 
            prefix ? nullptr : tab.signs(), 
            depth_level);
        LASTERR("failed to launch print_tableau_k kernel");
        SYNCALL;
        fflush(stdout);
	}

	void Simulator::print_gates(const DeviceCircuit& gpu_circuit, const gate_ref_t& num_gates, const depth_t& depth_level) {
		if (!options.print_gates) return;
		SYNCALL;
		LOG2(0, " Gates on GPU for %d-time step:", depth_level);
		print_gates_k << <1, 1 >> > (
            gpu_circuit.references(), 
            gpu_circuit.gates(), 
            num_gates);
		LASTERR("failed to launch print_gates_k kernel");
		SYNCALL;
		fflush(stdout);
	}
    void Simulator::print_signs(const Tableau& tab, const depth_t& depth_level) {
		if (!options.print_signs) return;
		SYNCALL;
		LOG2(0, " Signs on GPU for %d-time step:", depth_level);
		print_signs_k << <1, 1 >> > (
            tab.signs(), 
            depth_level);
		LASTERR("failed to launch print_signs_k kernel");
		SYNCALL;
		fflush(stdout);
	}

    void Simulator::print_progress_header() {
        LOGN2(1, "   %-10s    %-10s    %-10s    %15s          %-9s", 
                "Partition", "Step", "Gates", "Measurements", "Time (s)");
        if (options.check_tableau || options.check_measurement)
            LOG2(1, "  %s", "Integrity");
        else
            LOG2(1, "");
        LOGN2(1, "   %-10s    %-10s    %-10s    %-10s  %-10s    %-10s", 
                "", "", "", "definite", "random", "");
        if (options.check_tableau || options.check_measurement)
            LOG2(1, "  %s", "");
        else
            LOG2(1, "");
        LOGRULER(1, '-', RULERLEN);
    }

    void Simulator::print_progress(const size_t& p, const depth_t& depth_level, const bool& passed) {
        if (options.progress_en) {
            progress_timer.stop();
            const bool is_measuring = circuit.is_measuring(depth_level);
            size_t random_measures = stats.circuit.measure_stats.random_per_window;
            stats.circuit.measure_stats.random_per_window = 0;
            size_t prev_num_gates = circuit[depth_level].size();
            size_t definite_measures = is_measuring ? prev_num_gates - random_measures : 0;
            if (is_measuring) SETCOLOR(CLBLUE, stdout);
            else SETCOLOR(CORANGE1, stdout);
            LOGN2(1, "%c  %-10lld    %-10lld    %-10lld    %-10lld  %-10lld   %-7.3f", 
                    is_measuring ? 'm' : 'u',
                    p + 1, 
                    depth_level + 1, 
                    prev_num_gates, 
                    definite_measures, 
                    random_measures, 
                    progress_timer.elapsed() / 1000.0);
            if (options.check_tableau ||
                (options.check_measurement && is_measuring))
                LOG2(1, "    %s%-10s%s", 
                passed ? CGREEN : CRED,
                passed ? "PASSED" : "FAILED", CNORMAL);
            else
                LOG2(1, "");
            SETCOLOR(CNORMAL, stdout);
        }
    }

}

