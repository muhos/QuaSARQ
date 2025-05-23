
#include "equivalence.hpp"
#include "simulator.hpp"
#include "control.hpp"
#include "banner.hpp"
#include "input.hpp"
#include "tuner.cuh"
#include "logging.hpp"

using namespace std;
using namespace QuaSARQ;

int main(int argc, char** argv) {
    SETCOLOR(CNORMAL, stdout);

    if (argc == 1) LOGERROR("at least one argument is missing.");

	try {
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
			Equivalence* equivalence = has_input_file == 2 ? new Equivalence(string(argv[1]), string(argv[2])) : new Equivalence();
			signal_timeout(equivalence->handler_timeout);
			equivalence->check();
			LOGHEADER(1, 4, "Exit");
			delete equivalence;
		}
		else if (options.tuner_en) {
			Tuner* tuner = has_input_file ? new Tuner(string(argv[1])) : new Tuner();
			signal_timeout(tuner->handler_timeout);
			tuner->run();
			LOGHEADER(1, 4, "Exit");
			delete tuner;
		}
        else {
			Simulator* sim = has_input_file ? new Simulator(string(argv[1])) : new Simulator();
			signal_timeout(sim->handler_timeout);
            sim->simulate();
		    LOGHEADER(1, 4, "Exit");
			delete sim;
        }

        LOGRULER(1, '-', RULERLEN);
		return EXIT_SUCCESS;
	}
	catch (std::bad_alloc&) {
		CHECK(cudaDeviceReset());
		LOGERRORN("Emergency exit due to something not good, not good at all.");
		return EXIT_FAILURE;
	}
    catch (GPU_memory_exception&) {
		CHECK(cudaDeviceReset());
		LOGERRORN("Emergency exit due to GPU memory disaster.");
		return EXIT_FAILURE;
	}
	catch (CPU_memory_exception&) {
		CHECK(cudaDeviceReset());
		LOGERRORN("Emergency exit due to CPU memory disaster.");
		return EXIT_FAILURE;
	}
}