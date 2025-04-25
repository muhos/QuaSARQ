#pragma once

#include "malloc.hpp"
#include "gate.cuh"

namespace QuaSARQ {

    struct Gate_stats {

        size_t* types;

        Gate_stats() : types(nullptr) { }

        void alloc() {
            if (types == nullptr)
                types = calloc<size_t>(NR_GATETYPES);
        }

        void operator=(const Gate_stats& other) {
            for (uint32 i = 0; i < NR_GATETYPES; i++)
                types[i] = other.types[i];
        }

        void destroy() {
            if (types != nullptr) {
                std::free(types);
                types = nullptr;
            }
        }

        size_t all() {
            assert(types != nullptr);
            size_t all = 0;
            for (uint32 i = 0; i < NR_GATETYPES; i++)
                all += types[i];
            return all;
        }
    };

    struct Measures {
        size_t random, definite;
        size_t random_per_window;
    };

    struct Statistics {
        struct {
            Gate_stats gate_stats;
            Measures measure_stats;
            size_t bytes;
            size_t num_gates;
            size_t num_parallel_gates;
            size_t max_parallel_gates;
        } circuit;

        struct {
            double gigabytes;
            double seconds;
            double speed;
            size_t count;
            size_t istates;

            void calc_speed() {
                if (seconds <= 0)
                    speed = 0;
                speed = (istates * count * gigabytes) / seconds;
            }
        } tableau;

        struct {
            double initial;
            double schedule;
            double transfer;
            double simulation;

            double total() {
                return initial + schedule + transfer + simulation;
            }
        } time;

        struct {
            double wattage;
            double joules;
        } power;

        Statistics() {
            RESETSTRUCT(this);
            circuit.gate_stats.alloc();
        }

        ~Statistics() {
            circuit.gate_stats.destroy();
        }

        void reset() {
            circuit.gate_stats.destroy();
            RESETSTRUCT(this);
            circuit.gate_stats.alloc();
        }
    };

}