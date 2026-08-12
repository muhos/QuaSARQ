#include <exception>
#include <memory>
#include <new>
#include <stdexcept>

#include "sampler.hpp"
#include "module.hpp"
#include "simulate.hpp"
#include <nanobind/stl/vector.h>

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

    nb::class_<Module::Circuit>(m, "Circuit")
        .def(nb::init<nb::handle>(), nb::arg("circuit"),
             "Build a circuit from stim-format text, a stim.Circuit, or another Circuit.")
        .def_prop_ro("num_qubits", &Module::Circuit::num_qubits)
        .def_prop_ro("num_measurements", &Module::Circuit::num_measurements)
        .def_prop_ro("num_detectors", &Module::Circuit::num_detectors)
        .def_prop_ro("num_observables", &Module::Circuit::num_observables)
        .def("__str__", &Module::Circuit::circuit_text)
        .def("__repr__", [](const Module::Circuit& self) {
            return "quasarq.Circuit(" + std::to_string(self.num_qubits()) + " qubits, "
                 + std::to_string(self.num_measurements()) + " measurements, "
                 + std::to_string(self.num_detectors()) + " detectors, "
                 + std::to_string(self.num_observables()) + " observables)";
        });

    nb::class_<Simulation>(m, "Simulation")
        .def_prop_ro("num_qubits", &Simulation::qubits)
        .def_prop_ro("num_measurements", &Simulation::measurements)
        .def("measurements",
             [](Simulation& self) {
                 const std::vector<bool>& record = Module::guarded(
                     "simulation failed", [&]() -> const std::vector<bool>& {
                         return self.measurement_record();
                     });
                 Module::HostBuffer bits(record.size(), record.empty() ? 0 : 1);
                 for (size_t i = 0; i < record.size(); i++)
                     bits.data[i] = uint8_t(record[i]);
                 nb::object array = Module::to_numpy(bits, false);
                 return array.attr("reshape")(record.size());
             },
             "Outcome of every measurement in circuit order, as a bool array.")
        .def("paulis",
             [](Simulation& self) {
                 return Module::guarded("simulation failed", [&]() -> const std::vector<std::string>& {
                     return self.pauli_strings();
                 });
             },
             "The state the circuit ends in, one Pauli string per generator: a sign followed by "
             "one letter per qubit. An extended tableau lists the destabilizers first, then the "
             "stabilizers. These are rows of the inverse tableau, which is what this path holds.");

    m.def("equivalent",
          [](nb::handle circuit, nb::handle other) {
              std::string a = Module::circuit_to_text(circuit);
              std::string b = Module::circuit_to_text(other);
              return Module::guarded("equivalence check failed",
                                     [&] { return QuaSARQ::check_equivalence(a, b); });
          },
          nb::arg("circuit"), nb::arg("other"),
          "True when two circuits realize the same Clifford operation.");

    m.def("simulate",
          [](nb::handle circuit) {
              std::string text = Module::circuit_to_text(circuit);
              return Module::guarded("failed to parse the circuit",
                                     [&] { return new Simulation(text); });
          },
          nb::arg("circuit"),
          "Run a circuit once, deterministically, and report what it measured and the state it "
          "ended in. This is the single-shot simulation.");

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

    m.def("set_max_device_memory",
          [](nb::handle limit) {
              if (nb::isinstance<nb::str>(limit)) {
                  const std::string choice = nb::cast<std::string>(limit);
                  if (choice != "auto")
                      throw std::invalid_argument("the only accepted string is 'auto', got '" + choice + "'");
                  binding_set_auto_device_memory();
                  return;
              }
              binding_set_max_device_memory(nb::cast<int>(limit));
          },
          nb::arg("megabytes"),
          "Cap the GPU pool so several processes can share one device. Takes a size in MB, or "
          "'auto' to size it per run from the circuit and the shot count. 0 takes whatever is "
          "free. Must be set before the first sample().");

    m.def("get_max_device_memory", &binding_get_max_device_memory);

    m.def("set_kernel_config", [](const std::string& path) { binding_initialize(path); },
          nb::arg("path"), "Point the sampling engine at a kernel.config file.");

    m.def("version", &binding_version);

    m.def("device_name", &binding_device_name);
}
