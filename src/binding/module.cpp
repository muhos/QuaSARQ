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

nb::object Module::DetectorSampler::sample(
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
        execute(request, { &dets, &obs });
    }

    nb::object dets_array = to_numpy(dets, bit_packed);
    if (!separate_observables) {
        discard(obs);
        return dets_array;
    }
    nb::object obs_array = to_numpy(obs, bit_packed);
    return nb::make_tuple(dets_array, obs_array);
}

nb::object Module::MeasurementSampler::sample(const size_t& shots, const bool& bit_packed)
{
    const size_t num_measurements = engine->measurements();
    const size_t stride = binding_stride_of(num_measurements, bit_packed);

    HostBuffer measurements(shots, stride);

    if (shots) {
        SampleRequest request;
        request.num_shots = shots;
        request.bit_packed = bit_packed;
        request.measurements = measurements.data;
        request.measurements_stride = stride;
        execute(request, { &measurements });
    }

    return to_numpy(measurements, bit_packed);
}

NB_MODULE(_quasarq, m) {

    m.doc() = "QuaSARQ GPU simulator, compatible with Stim format.";

    binding_initialize(Module::locate_kernel_config());

    nb::class_<Module::DetectorSampler>(m, "CompiledDetectorSampler")
        .def_prop_ro("num_detectors", &Module::DetectorSampler::num_detectors)
        .def_prop_ro("num_observables", &Module::DetectorSampler::num_observables)
        .def_prop_ro("holds_device_memory", &Module::DetectorSampler::holds_device_memory)
        .def("release", &Module::DetectorSampler::release, "Free the GPU pool this sampler is holding. The next sample() rebuilds it.")
        .def("sample", &Module::DetectorSampler::sample,
             nb::arg("shots"),
             nb::kw_only(),
             nb::arg("separate_observables") = false,
             nb::arg("bit_packed") = false,
             nb::arg("append_observables") = false,
             nb::arg("prepend_observables") = false,
             "Sample detection events. Returns dets, or (dets, obs) when "
             "separate_observables=True.");

    nb::class_<Module::MeasurementSampler>(m, "CompiledMeasurementSampler")
        .def_prop_ro("num_measurements", &Module::MeasurementSampler::num_measurements)
        .def_prop_ro("holds_device_memory", &Module::MeasurementSampler::holds_device_memory)
        .def("release", &Module::MeasurementSampler::release, "Free the GPU pool this sampler is holding. The next sample() rebuilds it.")
        .def("sample", &Module::MeasurementSampler::sample,
             nb::arg("shots"),
             nb::kw_only(),
             nb::arg("bit_packed") = false,
             "Sample raw measurement outcomes, one column per measurement in circuit order.");

    m.def("compile_detector_sampler",
          [](nb::handle circuit, nb::handle seed) {
              const uint64_t resolved = seed.is_none() ? binding_random_seed() : nb::cast<uint64_t>(seed);
              return new Module::DetectorSampler(circuit, resolved);
          },
          nb::arg("circuit"),
          nb::kw_only(),
          nb::arg("seed") = nb::none(),
          "Compile a detection-event sampler for a stim.Circuit or circuit text. "
          "seed=None samples non-deterministically.");

    m.def("compile_sampler",
          [](nb::handle circuit, nb::handle seed) {
              const uint64_t resolved = seed.is_none() ? binding_random_seed() : nb::cast<uint64_t>(seed);
              return new Module::MeasurementSampler(circuit, resolved);
          },
          nb::arg("circuit"),
          nb::kw_only(),
          nb::arg("seed") = nb::none(),
          "Compile a measurement sampler for a stim.Circuit or circuit text. "
          "seed=None samples non-deterministically.");

    m.def("set_verbosity", &binding_set_verbosity, nb::arg("level"),
          "0 silences the sampling engine; 1-3 print progress to stdout.");

    m.def("get_verbosity", &binding_get_verbosity);

    m.def("set_chunk_shots", &binding_set_chunk_shots, nb::arg("shots"),
          "Cap how many shots are simulated per GPU chunk. 0 sizes the chunk to fit device memory.");

    m.def("get_chunk_shots", &binding_get_chunk_shots);

    m.def("set_kernel_config", [](const std::string& path) { binding_initialize(path); },
          nb::arg("path"), "Point the sampling engine at a kernel.config file.");

    m.def("version", &binding_version);

    m.def("device_name", &binding_device_name);
}
