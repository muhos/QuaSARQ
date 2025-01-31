
#include "options.hpp"
#include "input.hpp"
#include "macros.cuh"

namespace QuaSARQ {

    #define CONFIG2INPUT(CONFIG) \
        BOOL_OPT opt_tune_ ## CONFIG ## _en("tune-"#CONFIG, "tune "#CONFIG" kernels", false);

    #define CONFIG2ASSIGN(CONFIG) \
        tune_ ## CONFIG = opt_tune_ ## CONFIG ## _en;

    #define ENABLE_TUNER(CONFIG) \
        tuner_en |= tune_ ## CONFIG;

    // Default frequency of some gates will be changed later.
    #define GATE2INPUT(GATE) \
        DOUBLE_OPT opt_ ## GATE ## _prob(#GATE, "Frequency of " #GATE " gates in a generated random circuit", (1.0 / NR_GATETYPES), FP64R(0,1));

    #define GATE2ASSIGN(GATE) \
        GATE ## _p = opt_ ## GATE ## _prob;

    BOOL_OPT opt_quiet_en("q", "be quiet", false);
    BOOL_OPT opt_report_en("report", "report statistics", true);
    BOOL_OPT opt_progress_en("progress", "report progress", true);
    BOOL_OPT opt_equivalence_en("equivalence", "do equivalence checking", false);
    BOOL_OPT opt_checkparallelgates_en("check-parallel-gates", "check parallel gates independency", false);
    BOOL_OPT opt_checkintegrity_en("check-integrity", "check circuit integrity for possible logical errors", false);
    BOOL_OPT opt_print_tableau_step("print-step-tableau", "print tableau after every simulation step on screen in binary format", false);
    BOOL_OPT opt_print_tableau_final("print-final-tableau", "print final tableau after simulation ends on screen in binary format", false);
    BOOL_OPT opt_print_tableau_initial("print-initial-tableau", "print initial tableau before simulation on screen in binary format", false);
    BOOL_OPT opt_print_step_state("print-step-state", "print step state in form of Pauli strings on screen", false);
    BOOL_OPT opt_print_final_state("print-final-state", "print final state in form of Pauli strings on screen", false);
    BOOL_OPT opt_print_gates("print-gates", "print gates in every step on screen", false);
    BOOL_OPT opt_print_measurements("print-measurements", "print gates in every step on screen", false);
    BOOL_OPT opt_sync("sync", "synchronize all kernels and data transfers", false);
    BOOL_OPT opt_profile_equivalence("profile-equivalence", "profile equivalence checking", false);
    BOOL_OPT opt_disable_concurrency("disable-concurrency", "disable concurrency in equivalence checking", false);
    FOREACH_CONFIG(CONFIG2INPUT);

    INT_OPT opt_initialstate("initial", "set initial quantum state (0: 0 state, 1: + state, 2: i state)", 0, INT32R(0, 2));
    INT_OPT opt_streams("streams", "number of GPU streams to create", 4, INT32R(4, 32));
    INT_OPT opt_verbose("verbose", "set verbosity level", 1, INT32R(0, 3));
    INT_OPT opt_write_rc("write-circuit", "write generated circuit to file (1: stim, 2: chp)", 0, INT32R(0, 2));

    INT64_OPT opt_tuneinitial_qubits("tune-initial-qubits", "set the initial number of qubits to start with in the tuner", 1000, INT64R(1, UINT32_MAX));
    INT64_OPT opt_tunestep_qubits("tune-step-qubits", "set the increase of qubits", 1000, INT64R(1, UINT32_MAX));
    INT64_OPT opt_num_qubits("qubits", "set number of qubits for random generation (if no input file given)", 1000, INT64R(1, UINT32_MAX));
    INT64_OPT opt_depth("depth", "set circuit depth for random generation (if no input file given)", 2, INT64R(1, UINT32_MAX));

    FOREACH_GATE(GATE2INPUT);

    STRING_OPT opt_configpath("config-path", "Set the path of the kernel configuration file", "kernel.config");

    Options::Options() {
        RESETSTRUCT(this);
        configpath = calloc<char>(256);
    }

    Options::~Options() {
		if (configpath != nullptr) {
			std::free(configpath);
            configpath = nullptr;
		}
    }

    void Options::initialize() {
        quiet_en = opt_quiet_en;
        report_en = opt_report_en;
        progress_en = opt_progress_en;
        verbose = opt_verbose;
        if (options.quiet_en) options.verbose = 0;
        else if (!options.verbose) options.quiet_en = true;

        equivalence_en = opt_equivalence_en;
        profile_equivalence = opt_profile_equivalence;
        disable_concurrency = opt_disable_concurrency;

        check_parallel_gates = opt_checkparallelgates_en;
        check_integrity = opt_checkintegrity_en;
        checker_en = check_parallel_gates || check_integrity;

        sync = opt_sync;

        FOREACH_CONFIG(CONFIG2ASSIGN);
        FOREACH_CONFIG(ENABLE_TUNER);
        tuner_initial_qubits = opt_tuneinitial_qubits;
        tuner_step_qubits = opt_tunestep_qubits;

        print_final_state = opt_print_final_state;
        print_step_state = opt_print_step_state;
        print_gates = opt_print_gates;
        print_measurements= opt_print_measurements;
        print_step_tableau = opt_print_tableau_step;
        print_final_tableau = opt_print_tableau_final;
        print_initial_tableau = opt_print_tableau_initial;
        write_rc = opt_write_rc;

        opt_H_prob = 0.08;
        opt_S_prob = 0.08;
        opt_CX_prob = 0.09;
        opt_M_prob = 0.09;
        FOREACH_GATE(GATE2ASSIGN);

        initialstate = InitialState(int(opt_initialstate));
        num_qubits = opt_num_qubits;
        depth = opt_depth;
        streams = opt_streams;

        std::memcpy(configpath, opt_configpath, opt_configpath.length());
    }

    void Options::check(const char* inpath, const char* other_inpath) {
        // mixing options handling
        if (other_inpath != nullptr) {
            if (!equivalence_en) {
                LOG2(1, "%s  Turning on equivalence checking of other circuit.%s", CARGDEFAULT, CNORMAL);
                equivalence_en = 1;
            }
        }
        if (equivalence_en) {
            tuner_en = false;
            checker_en = false;
        }
        if (tuner_en && tuner_step_qubits > num_qubits) {
            LOG2(1, "%s  Stepwise qubits %s%zd%s is downsized to %s%zd%s maximum.%s", CARGDEFAULT, CARGVALUE, tuner_step_qubits, CARGDEFAULT, CARGVALUE, num_qubits, CARGDEFAULT, CNORMAL);
            depth = 1;
        }
        if (inpath != nullptr && num_qubits > 1) {
            LOGWARNING("entered number of qubits will be overridden by the circuit file.");
        }
        if (inpath != nullptr && depth > 1) {
            LOGWARNING("entered depth will be overridden by the circuit file.");
        }
        if (inpath != nullptr && !hasstr(inpath, ".stim") && !hasstr(inpath, ".qasm")) {
            LOGERROR("file \"%s\" has unsupported file format.", inpath);
        }
        if (other_inpath != nullptr && !hasstr(other_inpath, ".stim") && !hasstr(other_inpath, ".qasm")) {
            LOGERROR("file \"%s\" has unsupported file format.", other_inpath);
        }
        if (equivalence_en && inpath != nullptr && other_inpath == nullptr) {
            LOGERROR("missing other citcuit to check.");
        }
    }

    Options options;

}