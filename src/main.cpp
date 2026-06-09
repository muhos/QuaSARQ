
#include "equivalence.hpp"
#include "simulator.hpp"
#include "control.hpp"
#include "logging.hpp"
#include "banner.hpp"
#include "input.hpp"
#include "frame.cuh"
#include "tuner.cuh"

#include <memory>

using namespace std;
using namespace QuaSARQ;

int main(int argc, char** argv) {
    SETCOLOR(CNORMAL, stdout);

	try {
        if (argc == 1) LOGERROR("at least one argument is missing.");

		int has_input_file = parseArguments(argc, argv);
        options.initialize();
		SET_LOGGER_VERBOSITY(options.verbose);
		if (!options.quiet_en && options.verbose) {
			LOGHEADER(1, 4, "Banner");
			LOGFANCYBANNER(version());
			printArguments(argc - 1 > has_input_file);
		}
		options.check(has_input_file == 1 ? argv[1] : nullptr, has_input_file == 2 ? argv[2] : nullptr);

		signal_termination(handler_terminate);

		if (options.timeout > 0) set_timeout(options.timeout);

        if (options.equivalence_en) {
			auto equivalence = has_input_file == 2 ? make_unique<Equivalence>(string(argv[1]), string(argv[2])) : make_unique<Equivalence>();
			signal_timeout(equivalence->handler_timeout);
			equivalence->check();
			LOGHEADER(1, 4, "Exit");
		}
		else if (options.num_shots) {
			auto framing = has_input_file ? make_unique<Framing>(string(argv[1]), options.num_shots) : make_unique<Framing>(options.num_shots);
			signal_timeout(framing->handler_timeout);
			framing->sample();
			LOGHEADER(1, 4, "Exit");
		}
		else if (options.tuner_en) {
			auto tuner = has_input_file ? make_unique<Tuner>(string(argv[1])) : make_unique<Tuner>();
			signal_timeout(tuner->handler_timeout);
			tuner->run();
			LOGHEADER(1, 4, "Exit");
		}
        else {
			auto sim = has_input_file ? make_unique<Simulator>(string(argv[1])) : make_unique<Simulator>();
			signal_timeout(sim->handler_timeout);
            sim->simulate();
		    LOGHEADER(1, 4, "Exit");
        }

		return EXIT_SUCCESS;
	}
	catch (fatal_error&) {
		cudaDeviceReset();
		return EXIT_FAILURE;
	}
	catch (std::bad_alloc&) {
		cudaDeviceReset();
		LOGERRORN("Emergency exit due to something not good, not good at all.");
		return EXIT_FAILURE;
	}
    catch (tableau_memory_error&) {
		cudaDeviceReset();
		LOGERRORN("Emergency exit due to GPU memory disaster.");
		return EXIT_FAILURE;
	}
	catch (malloc_memory_error&) {
		cudaDeviceReset();
		LOGERRORN("Emergency exit due to CPU memory disaster.");
		return EXIT_FAILURE;
	}
	catch (cuArena::gpu_memory_error&) {
		cudaDeviceReset();
		LOGERRORN("Emergency exit due to GPU memory disaster.");
		return EXIT_FAILURE;
	}
	catch (cuArena::cpu_memory_error&) {
		cudaDeviceReset();
		LOGERRORN("Emergency exit due to CPU memory disaster.");
		return EXIT_FAILURE;
	}
	catch (std::exception& err) {
		cudaDeviceReset();
		LOGERRORN("Emergency exit due to exception: %s", err.what());
		return EXIT_FAILURE;
	}
	catch (...) {
		cudaDeviceReset();
		LOGERRORN("Emergency exit due to unknown exception.");
		return EXIT_FAILURE;
	}
}
