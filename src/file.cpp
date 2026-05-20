#include "simulator.hpp"

using namespace QuaSARQ;

bool Simulator::open_file(FILE*& file, arg_t file_path, arg_t file_mode) {
    if (file == nullptr) {
        LOGN2(1, "Opening \"%s%s%s\" file for \"%s%s%s\".. ", 
            CREPORTVAL, file_path, CNORMAL, 
			CREPORTVAL, hasstr(file_mode, "r") ? "reading" : "writing", CNORMAL);
        file = fopen(file_path, file_mode);
        if (file == nullptr) { 
			LOG2(1, "does not exist."); 
			return false; 
		}
		LOGDONE(1, 4);
    }
	return true;
}

void Simulator::close_file(FILE*& file) {
    if (file != nullptr) {
        fclose(file);
        file = nullptr;
    }
}

FILE* Simulator::open_output_file(const string& suffix) {
    string base = circuit_path;
    const size_t dot = base.rfind('.');
    if (dot != string::npos) base = base.substr(0, dot);
    if (base.empty()) base = "quasarq";
    const string path = base + suffix;
    FILE* f = fopen(path.c_str(), "w");
    if (f == nullptr)
        LOGERROR("failed to open output file \"%s\"", path.c_str());
    LOG2(1, " %sWriting to \"%s%s%s\".%s", CREPORT, CREPORTVAL, path.c_str(), CREPORT, CNORMAL);
    return f;
}