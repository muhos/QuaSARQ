#pragma once

#include <nvml.h>
#include "datatypes.hpp"

namespace QuaSARQ {

    class Power {

        // Initial power draw reading from 
        // nvidia-smi (in milliwatt).
        static constexpr uint32 INITPWR = 34 * 1000;

        nvmlDevice_t device;
        uint32 power;

        public:

        Power() : device(0), power(0) {
            nvmlInit();
            nvmlDeviceGetHandleByIndex(0, &device);
        }
        ~Power() {
            nvmlShutdown();
        }

        // Measure power in wattage.
        double measure() {
            nvmlDeviceGetPowerUsage(device, &power);
            if (power >= INITPWR) power -= INITPWR;
            return double(power) / 1000.0;
        }

    };

}