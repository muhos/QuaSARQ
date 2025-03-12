
#include "simulator.hpp"

namespace QuaSARQ {

	FOREACH_CONFIG_INIT(CONFIG2INITIAL);

	#define CONFIG2BOOL(NAME) \
		bool fetched ## NAME = false;

	#define CONFIG2FETCHED(CONFIG) \
		dim3 bestblock ## CONFIG ## Fetched; \
		dim3 bestgrid ## CONFIG ## Fetched;

	#define CONFIG2FETCHED_LOGIC(NAME) \
	{ \
		size_t count = 0; \
		if (count = hasstr(line, #NAME)) { \
			line += count; \
			bestgrid ## NAME ## Fetched.x = toInteger(line); \
			bestgrid ## NAME ## Fetched.y = toInteger(line); \
			bestblock ## NAME ## Fetched.x = toInteger(line); \
			bestblock ## NAME ## Fetched.y = toInteger(line); \
			fetched ## NAME = true; \
			eatWS(line); \
		} \
	}

	#define CONFIG2APPLY(CONFIG) \
		if (fetched ## CONFIG) { \
			bestblock ## CONFIG = bestblock ## CONFIG ## Fetched; \
			bestgrid ## CONFIG = bestgrid ## CONFIG ## Fetched; \
		}

	#define CONFIG2PRINT(CONFIG) \
		if (fetched ## CONFIG) { \
			LOG2(1, " read " #CONFIG " configuration with %lld distance: grid(%d, %d), block(%d, %d)", min_diff, bestgrid ## CONFIG.x, bestgrid ## CONFIG.y, bestblock ## CONFIG.x, bestblock ## CONFIG.y); \
		}
}

using namespace QuaSARQ;

bool Simulator::open_config(arg_t file_mode) {
    if (config_file == nullptr) {
        LOGN2(1, "Opening \"%s%s%s\" kernel configuration file for %s.. ", CREPORTVAL, options.configpath, CNORMAL, hasstr(file_mode, "r") ? "reading" : "writing");
        config_file = fopen(options.configpath, file_mode);
        if (config_file == nullptr) { 
			LOG2(1, "does not exist."); 
			return false; 
		}
		LOGDONE(1, 3);
    }
	return true;
}

void Simulator::close_config() {
    if (config_file != nullptr) {
        fclose(config_file);
        config_file = nullptr;
    }
}

void Simulator::register_config() {
    if (!open_config()) {
		LOGERROR("cannot proceed without registering kernel configuration.");
	}
    struct stat st;
	if (!canAccess(options.configpath, st)) {
		LOGERROR("kernel configuration file is inaccessible.");
	}
	if (!st.st_size) return;
	assert(st.st_size < UINT32_MAX);
    char* buffer = calloc<char>(st.st_size);
	if (!fread(buffer, 1, st.st_size, config_file))
		LOGERROR("cannot read kernel configuration file.");
	int64 min_diff = UINT32_MAX;
    config_qubits = 0;
	FOREACH_CONFIG(CONFIG2FETCHED);
	FOREACH_CONFIG(CONFIG2BOOL);
	char* line = buffer;
    while (line[0] != '\0') {
		if (line[0] == '\001' || line[0] == '\003') {
			eatLine(line);
			continue;
		} 
        config_qubits = toInteger(line);
        if (!config_qubits)
            LOGERROR("Expected non-zero number of qubits."); 
		int64 diff = abs(int64(num_qubits) - config_qubits);
		assert(diff >= 0);
		eatWS(line);
		FOREACH_CONFIG(CONFIG2FETCHED_LOGIC);
		if (diff <= min_diff) {
			min_diff = diff;
			FOREACH_CONFIG(CONFIG2APPLY);
		}
    }
	FOREACH_CONFIG(CONFIG2PRINT);
    close_config();
    std::free(buffer);
}