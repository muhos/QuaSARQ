#pragma once

#include <nvml.h>
#include "datatypes.hpp"

namespace QuaSARQ {

    class Power {

        // Initial power draw reading from 
        // nvidia-smi (in milliwatt).
        static constexpr uint32 INITPWR = 34 * 1000;

        uint32 power;

        // Process-wide NVML device handle, lazily opened on first use. 
        // Return 0 if NVML fails to initialize or the device can't be opened.
        static nvmlDevice_t handle() {
            static nvmlDevice_t device = []() {
                nvmlDevice_t opened = 0;
                if (nvmlInit() != NVML_SUCCESS) return nvmlDevice_t(0);
                if (nvmlDeviceGetHandleByIndex(0, &opened) != NVML_SUCCESS) return nvmlDevice_t(0);
                return opened;
            }();
            return device;
        }

        public:

        Power() : power(0) {}

        // Measure power in wattage.
        double measure() {
            nvmlDevice_t device = handle();
            if (device == nvmlDevice_t(0)) return 0.0;
            if (nvmlDeviceGetPowerUsage(device, &power) != NVML_SUCCESS) return 0.0;
            if (power >= INITPWR) power -= INITPWR;
            return double(power) / 1000.0;
        }

    };

}