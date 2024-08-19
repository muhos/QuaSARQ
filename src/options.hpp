
#ifndef __OPTIONS_H
#define __OPTIONS_H

#include "datatypes.hpp"
#include "constants.hpp"

namespace QuaSARQ {

    struct Options {
        
        int verbose;
        int streams;

        bool quiet_en;
        bool report_en;
        bool overlap;
        bool sync;

        bool equivalence_en;
        
        bool checker_en;
        bool check_parallel_gates;
        bool check_integrity;

        bool tuner_en;
        bool tune_identity;
        bool tune_step;

        bool write_rc;
        bool print_gates;
        bool print_step_tableau;
        bool print_final_tableau;
        bool print_initial_tableau;
        bool print_step_state;
        bool print_final_state;

        bool profile_equivalence;
        bool disable_concurrency;

        size_t tuner_initial_qubits;
        size_t tuner_step_qubits;
        size_t num_qubits;
        size_t depth;

        double H_p, S_p, CX_p;

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