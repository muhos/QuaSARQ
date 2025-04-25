#pragma once

#include "datatypes.hpp"
#include "constants.hpp"
#include "kernelconfig.hpp"
#include "gatetypes.hpp"

namespace QuaSARQ {

    #define FOREACH_CHECK(CONFIG) \
        CONFIG(scheduler) \
        CONFIG(identity) \
        CONFIG(tableau) \
        CONFIG(transpose) \
        CONFIG(measurement) \

    #define FOREACH_PRINT(CONFIG) \
        CONFIG(gates, "gates") \
        CONFIG(measurements, "measurements") \
        CONFIG(initialstate, "initial state (in Pauli strings)") \
        CONFIG(stepstate, "step state (in Pauli strings)") \
        CONFIG(finalstate, "final state (in Pauli strings)") \
        CONFIG(initialtableau, "initial tableau (in binary)") \
        CONFIG(steptableau, "step tableau (in binary)") \
        CONFIG(finaltableau, "final tableau (in binary)") \

    #define CONFIG2PRINTOPTION(CONFIG, MSG) \
        bool print_ ## CONFIG;

    #define CONFIG2TUNEOPTION(CONFIG, BLOCKX, BLOCKY, GRIDX, GRIDY) \
        bool tune_ ## CONFIG;

    #define CONFIG2CHECKOPTION(CONFIG) \
        bool check_ ## CONFIG;

    #define GATE2OPTION(GATE) \
        double GATE ## _p;
    
    struct Options {
        
        int verbose;
        int streams;
        int write_rc;

        bool quiet_en;
        bool report_en;
        bool progress_en;

        bool check_all;
        bool tuner_en;
        bool tune_all;
        bool tune_measurement;
        FOREACH_CONFIG(CONFIG2TUNEOPTION);
        FOREACH_CHECK(CONFIG2CHECKOPTION);
        FOREACH_PRINT(CONFIG2PRINTOPTION);

        bool equivalence_en;
        bool profile_equivalence;
        bool disable_concurrency;
        bool sync;

        size_t tuner_initial_qubits;
        size_t tuner_step_qubits;
        size_t num_qubits;
        size_t depth;

        FOREACH_GATE(GATE2OPTION);

        InitialState initialstate;

        char* configpath;

        Options();
        ~Options();

        void initialize();
        void check(const char* inpath, const char* other_inpath = nullptr);
    };

    extern Options options;

}