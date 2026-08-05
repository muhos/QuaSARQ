#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

#include "sampler.hpp"
#include "module.hpp"

using namespace QuaSARQ;

std::string Module::locate_kernel_config() {
    static const char* relatives[] = { "kernel.config", "../kernel.config" };
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&locate_kernel_config), &info) && info.dli_fname != nullptr) {
        const std::string self(info.dli_fname);
        const size_t slash = self.find_last_of('/');
        if (slash != std::string::npos) {
            const std::string dir = self.substr(0, slash + 1);
            for (const char* relative : relatives) {
                const std::string candidate = dir + relative;
                struct stat st;
                if (stat(candidate.c_str(), &st) == 0)
                    return candidate;
            }
        }
    }
    struct stat st;
    if (stat("src/kernel.config", &st) == 0)
        return std::string("src/kernel.config");
    return std::string();
}

nb::object Module::Sampler::sample(
                    const size_t& shots,
                    const bool& separate_observables,
                    const bool& bit_packed,
                    const bool& append_observables,
                    const bool& prepend_observables)
{
    if (append_observables || prepend_observables) {
        throw std::invalid_argument(
            "append_observables and prepend_observables are not supported; "
            "use separate_observables=True");
    }
    const size_t num_dets = engine->detectors();
    const size_t num_obs = engine->observables();
    const size_t dets_stride = binding_stride_of(num_dets, bit_packed);
    const size_t obs_stride = binding_stride_of(num_obs, bit_packed);

    HostBuffer dets(shots, dets_stride);
    HostBuffer obs(shots, separate_observables ? obs_stride : 0);

    if (shots) {
        SampleRequest request;
        request.num_shots = shots;
        request.bit_packed = bit_packed;
        request.detectors = dets.data;
        request.observables = separate_observables ? obs.data : nullptr;
        request.detectors_stride = dets_stride;
        request.observables_stride = obs_stride;
        binding_clear_error();
        std::lock_guard<std::mutex> lock(sampling_mutex);
        bool failed = false;
        {
            nb::gil_scoped_release release;
            try {
                engine->run(request);
            }
            catch (...) {
                failed = true;
            }
        }
        if (failed) {
            discard(dets);
            discard(obs);
            raise_from_core("sampling failed");
        }
    }

    nb::object dets_array = to_numpy(dets, bit_packed);
    if (!separate_observables) {
        discard(obs);
        return dets_array;
    }
    nb::object obs_array = to_numpy(obs, bit_packed);
    return nb::make_tuple(dets_array, obs_array);
}

NB_MODULE(quasarq, m) {

    m.doc() = "QuaSARQ GPU simulator, compatible with Stim format.";

    binding_initialize(Module::locate_kernel_config());

    nb::class_<Module::Sampler>(m, "CompiledSampler")
        .def_prop_ro("num_detectors", &Module::Sampler::num_detectors)
        .def_prop_ro("num_observables", &Module::Sampler::num_observables)
        .def_prop_ro("holds_device_memory", &Module::Sampler::holds_device_memory)
        .def("release", &Module::Sampler::release, "Free the GPU pool this sampler is holding. The next sample() rebuilds it.")
        .def("sample", &Module::Sampler::sample,
             nb::arg("shots"),
             nb::kw_only(),
             nb::arg("separate_observables") = false,
             nb::arg("bit_packed") = false,
             nb::arg("append_observables") = false,
             nb::arg("prepend_observables") = false,
             "Sample detection events. Returns dets, or (dets, obs) when "
             "separate_observables=True.");

    m.def("compile_sampler",
          [](nb::handle circuit, nb::handle seed) {
              const uint64_t resolved = seed.is_none() ? binding_random_seed() : nb::cast<uint64_t>(seed);
              return new Module::Sampler(circuit, resolved);
          },
          nb::arg("circuit"),
          nb::kw_only(),
          nb::arg("seed") = nb::none(),
          "Compile a sampler for a stim.Circuit or circuit text. "
          "seed=None samples non-deterministically, as in stim.");

    m.def("set_verbosity", &binding_set_verbosity, nb::arg("level"),
          "0 silences the sampling engine; 1-3 print progress to stdout.");

    m.def("get_verbosity", &binding_get_verbosity);

    m.def("set_kernel_config", [](const std::string& path) { binding_initialize(path); },
          nb::arg("path"), "Point the sampling engine at a kernel.config file.");

    m.def("version", &binding_version);

    m.def("device_name", &binding_device_name);
}
