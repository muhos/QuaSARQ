
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
        bool print_tableau_step;
        bool print_tableau_final;
        bool print_tableau_initial;
        bool print_stab;

        size_t tuner_initial_qubits;
        size_t tuner_step_qubits;
        size_t num_qubits;
        size_t depth;

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