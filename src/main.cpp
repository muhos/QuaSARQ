
#include "equivalence.hpp"
#include "simulator.hpp"
#include "banner.hpp"
#include "tuner.cuh"
#include "input.hpp"

using namespace std;
using namespace QuaSARQ;

int main(int argc, char** argv) {
    SETCOLOR(CNORMAL, stdout);

    if (argc == 1) LOGERROR("at least one argument is missing.");

	try {
		int has_input_file = parseArguments(argc, argv);
        options.initialize();
		if (!options.quiet_en && options.verbose) {
			LOGHEADER(1, 4, "Banner");
			LOGFANCYBANNER(version());
			printArguments(argc - 1 > has_input_file);
		}
		options.check(has_input_file == 1 ? argv[1] : nullptr, has_input_file == 2 ? argv[2] : nullptr);

		signal_handler(handler_terminate);

        if (options.equivalence_en) {
			Equivalence* equivalence = has_input_file == 2 ? new Equivalence(string(argv[1]), string(argv[2])) : new Equivalence();
			equivalence->check();
			LOGHEADER(0, 4, "Exit");
			delete equivalence;
		}
		else if (options.tuner_en) {
			Tuner* tuner = has_input_file ? new Tuner(string(argv[1])) : new Tuner();
			tuner->run();
			LOGHEADER(0, 4, "Exit");
			delete tuner;
		}
        else {
			Simulator* sim = has_input_file ? new Simulator(string(argv[1])) : new Simulator();
            sim->simulate();
		    LOGHEADER(0, 4, "Exit");
			delete sim;
        }

        if (options.verbose) LOGRULER('-', RULERLEN);
		return EXIT_SUCCESS;
	}
	catch (std::bad_alloc&) {
		CHECK(cudaDeviceReset());
		PRINT("%s%s%s", CERROR, "Emergency exit due to something not good, not good at all.\n", CNORMAL);
		return EXIT_FAILURE;
	}
    catch (GPU_memory_exception&) {
		CHECK(cudaDeviceReset());
		PRINT("%s%s%s", CERROR, "Emergency exit due to GPU memory disaster.\n", CNORMAL);
		return EXIT_FAILURE;
	}
	catch (CPU_memory_exception&) {
		CHECK(cudaDeviceReset());
		PRINT("%s%s%s", CERROR, "Emergency exit due to CPU memory disaster.\n", CNORMAL);
		return EXIT_FAILURE;
	}
}