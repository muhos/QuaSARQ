
#ifndef __KERNELCONFIG_H
#define __KERNELCONFIG_H

#include "datatypes.hpp"


namespace QuaSARQ {

    constexpr uint32 NSAMPLES_CONFIG = 1;

    // ~10000 samples for 1D block and grid.
    constexpr uint32 IDENTITY_CONFIG[NSAMPLES_CONFIG * 2] = { 0 };

    // ~10000 samples for 2D block and grid.
    constexpr uint32 STEP_CONFIG[NSAMPLES_CONFIG * 4] = { 0 };

}

#endif