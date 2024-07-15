
#include "simulator.hpp"
#include "grid.cuh"

using namespace QuaSARQ;

bool Simulator::open_config(arg_t file_mode) {
    if (configfile == nullptr) {
        LOGN2(1, "Opening \"%s%s%s\" kernel configuration file..", CREPORTVAL, options.configpath, CNORMAL);
        configfile = fopen(options.configpath, file_mode);
        if (configfile == nullptr) { 
			LOG2(1, "does not exist."); 
			return false; 
		}
		LOGDONE(1, 3);
    }
	return true;
}

void Simulator::close_config() {
    if (configfile != nullptr) {
        fclose(configfile);
        configfile = nullptr;
    }
}

void Simulator::register_config() {
    // No need to read configuration file.
    // if (IDENTITY_CONFIG[0] || STEP_CONFIG[0]) return;
    if (!open_config()) return;
    struct stat st;
	if (!canAccess(options.configpath, st)) {
		LOG2(1, "kernel configuration file is inaccessible.");
		return;
	}
	if (!st.st_size) return;
	assert(st.st_size < UINT32_MAX);
    char* buffer = calloc<char>(st.st_size);
	if (!fread(buffer, 1, st.st_size, configfile))
		LOGERROR("cannot read kernel configuration file.");
    // scan buffer and store configs in a map <key: qubits, value: config>
    // Print the read data
	int64 min_diff = UINT32_MAX;
    int64 config_qubits = 0;
	dim3 bestBlockResetFetched, bestGridResetFetched;
	dim3 bestBlockIdentityFetched, bestGridIdentityFetched;
	dim3 bestBlockStepFetched, bestGridStepFetched;
	bool fetchedR = false, fetchedI = false, fetchedS = false;
	char* line = buffer;
    while (line[0] != '\0') {
        //printf("%s\n", line);
        config_qubits = toInteger(line);
        if (!config_qubits)
            LOGERROR("Expected non-zero number of qubits."); 
		int64 diff = abs(int64(num_qubits) - config_qubits);
		assert(diff >= 0);
		eatWS(line);
		// Should be an I or S.
		assert(*line == 'R' || *line == 'I' || *line == 'S');
		if (*line == 'R') {
			bestGridResetFetched.x = toInteger(++line);
			if (!bestGridResetFetched.x)
				LOGERROR("Expected non-zero grid dimention for reset.");
			bestBlockResetFetched.x = toInteger(line);
			if (!bestBlockResetFetched.x)
				LOGERROR("Expected non-zero block dimention for reset.");
			fetchedR = true;
			eatWS(line);
		}
		if (*line == 'I') {
			bestGridIdentityFetched.x = toInteger(++line);
			if (!bestGridIdentityFetched.x)
				LOGERROR("Expected non-zero grid dimention for indentity.");
			bestBlockIdentityFetched.x = toInteger(line);
			if (!bestBlockIdentityFetched.x)
				LOGERROR("Expected non-zero block dimention for indentity.");
			fetchedI = true;
			eatWS(line);
		}
		if (*line == 'S') {
			bestGridStepFetched.x = toInteger(++line);
			if (!bestGridStepFetched.x)
				LOGERROR("Expected non-zero grid x-dimention for step.");
			bestGridStepFetched.y = toInteger(line);
			if (!bestGridStepFetched.y)
				LOGERROR("Expected non-zero grid y-dimention for step.");
			bestBlockStepFetched.x = toInteger(line);
			if (!bestBlockStepFetched.x)
				LOGERROR("Expected non-zero block x-dimention for step.");
			bestBlockStepFetched.y = toInteger(line);
			if (!bestBlockStepFetched.y)
				LOGERROR("Expected non-zero block y-dimention for step.");
			fetchedS = true;
			eatWS(line);
		}
		if (diff <= min_diff) {
			min_diff = diff;
			if (fetchedR) {
				bestBlockReset = bestBlockResetFetched;
				bestGridReset = bestGridResetFetched;
			}
			if (fetchedI) {
				bestBlockIdentity = bestBlockIdentityFetched;
				bestGridIdentity = bestGridIdentityFetched;
			}
			if (fetchedS) {
				bestBlockStep = bestBlockStepFetched;
				bestGridStep  = bestGridStepFetched;
			}
		}
		if (!fetchedI && !fetchedR && !fetchedS)
			LOGERROR("Expected at least one tuned kernel.");
    }
	double accuracy = 100.0 - percent(double(min_diff), double(num_qubits));
	if (fetchedR)
		LOG2(1, "Read best reset configuration with %%%.1f accuracy: grid(%d), block(%d)", accuracy, bestGridReset.x, bestBlockReset.x);
	if (fetchedI)
		LOG2(1, "Read best identity configuration with %%%.1f accuracy: grid(%d), block(%d)", accuracy, bestGridIdentity.x, bestBlockIdentity.x);
	if (fetchedS) {
		LOG2(1, "Read best step configuration with %%%.1f accuracy: grid(%d, %d), block(%d, %d)", accuracy, 
			bestGridStep.x, bestGridStep.y, bestBlockStep.x, bestBlockStep.y);
	}
    close_config();
    std::free(buffer);
}