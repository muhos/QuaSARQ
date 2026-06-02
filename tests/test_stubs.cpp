#include "simulator.hpp"

namespace QuaSARQ {

    Options::Options() {
        RESETSTRUCT(this);
        configpath = calloc<char>(256);
        statepath = calloc<char>(256);
    }

    Options::~Options() {
        if (configpath != nullptr) {
            std::free(configpath);
            configpath = nullptr;
        }
        if (statepath != nullptr) {
            std::free(statepath);
            statepath = nullptr;
        }
    }

    Options options;
    Timer timer;

}
