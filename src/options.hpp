
#ifndef __OPTIONS_H
#define __OPTIONS_H

#include "datatypes.hpp"
#include "constants.hpp"
#include "kernelconfig.hpp"

namespace QuaSARQ {

    #define CONFIG2OPTION(CONFIG) \
        bool tune_ ## CONFIG;
    
    struct Options {
        
        int verbose;
        int streams;
        int write_rc;

        bool quiet_en;
        bool report_en;
        bool progress_en;

        bool equivalence_en;
        bool checker_en;
        bool check_parallel_gates;
        bool check_integrity;

        bool tuner_en;
        FOREACH_CONFIG(CONFIG2OPTION);

        bool print_gates;
        bool print_step_tableau;
        bool print_final_tableau;
        bool print_initial_tableau;
        bool print_step_state;
        bool print_final_state;
        bool print_measurements;
        bool print_tableau_decimal;

        bool profile_equivalence;
        bool disable_concurrency;
        bool sync;

        size_t tuner_initial_qubits;
        size_t tuner_step_qubits;
        size_t num_qubits;
        size_t depth;

        double I_p, H_p, S_p, CX_p, M_p;

        InitialState initialstate;

        char* configpath;

        Options();
        ~Options();

        void initialize();
        void check(const char* inpath, const char* other_inpath = nullptr);
    };

    extern Options options;

}

#endif