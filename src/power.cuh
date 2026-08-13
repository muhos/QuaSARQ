#pragma once

#include <dlfcn.h>
#include <nvml.h>
#include "datatypes.hpp"

namespace QuaSARQ {

    class Power {

        // Initial power draw reading from 
        // nvidia-smi (in milliwatt).
        static constexpr uint32 INITPWR = 34 * 1000;

        uint32 power;

        // NVML ships with the driver, not with the toolkit, so it is loaded at run time rather
        // than linked during the build.
        struct NVML {
            nvmlReturn_t (*get_power)(nvmlDevice_t, uint32*);
            nvmlDevice_t device;
        };

        static const NVML& nvml() {
            static NVML loaded = []() {
                NVML api = {};
                void* library = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
                if (library == nullptr) return api;
                auto init = (nvmlReturn_t (*)()) dlsym(library, "nvmlInit_v2");
                auto by_index = (nvmlReturn_t (*)(uint32, nvmlDevice_t*)) dlsym(library, "nvmlDeviceGetHandleByIndex_v2");
                auto power_usage = (nvmlReturn_t (*)(nvmlDevice_t, uint32*)) dlsym(library, "nvmlDeviceGetPowerUsage");
                if (init == nullptr || by_index == nullptr || power_usage == nullptr) return api;
                nvmlDevice_t opened = 0;
                if (init() != NVML_SUCCESS) return api;
                if (by_index(0, &opened) != NVML_SUCCESS) return api;
                api.get_power = power_usage;
                api.device = opened;
                return api;
            }();
            return loaded;
        }

        public:

        Power() : power(0) {}

        // Measure power in wattage.
        double measure() {
            const NVML& api = nvml();
            if (api.device == nvmlDevice_t(0)) return 0.0;
            if (api.get_power(api.device, &power) != NVML_SUCCESS) return 0.0;
            if (power >= INITPWR) power -= INITPWR;
            return double(power) / 1000.0;
        }

    };

}