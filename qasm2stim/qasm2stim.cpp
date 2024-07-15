/*
	Muhammad Osama Mahmoud, 2024.
	A tool to convert OpenQASM v2 to Stim format.
	Parallel Quantum Project.
*/

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <sys/stat.h>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#elif defined(__CYGWIN__)
#include </usr/include/sys/resource.h>
#include </usr/include/sys/mman.h>
#include </usr/include/sys/sysinfo.h>
#include </usr/include/sys/unistd.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <intrin.h>
#include <Winnt.h>
#include <io.h>
#endif
#undef ERROR
#undef hyper 
#undef SET_BOUNDS
using std::string;
using std::ifstream;

namespace fs = std::filesystem;

class Timer {
    std::chrono::steady_clock::time_point _start, _end;
public:
    inline void  start() { _start = std::chrono::steady_clock::now(); }
    inline void  stop() { _end = std::chrono::steady_clock::now(); }
    inline double time() {
        return double(std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count());
    }
};


constexpr size_t MB = 0x00100000;
constexpr double ratio(const double& x, const double& y) { return y ? x / y : 0; }
constexpr size_t ratio(const size_t & x, const size_t & y) { return y ? x / y : 0; }

#define LOGERROR(FORMAT, ...) \
  do { \
     fprintf(stderr, "ERROR: " FORMAT "\n", ## __VA_ARGS__); \
     exit(1); \
  } while (0)

#define LOG(FORMAT, ...) \
  do { \
     fprintf(stdout, FORMAT, ## __VA_ARGS__); \
  } while (0)

inline bool isDigit(const char& ch) { return (ch ^ 48) <= 9; }

inline bool isSpace(const char& ch) { return (ch >= 9 && ch <= 13) || ch == 32; }

inline void eatWS(char*& str) { while (isSpace(*str)) str++; }

inline void eatLine(char*& str) { while (*str) if (*str++ == '\n') return; }

inline double toFloat(char*& str)
{
	eatWS(str);
	if (!isDigit(*str)) LOGERROR("expected a digit but ASCII(%d) is found", *str);
	double n = 0, f = 1;
    bool is_digit = false, is_point = false;
    char ch = *str;
    while (ch != ';') {
        is_digit = isDigit(ch);
        if (is_point) f /= 10.0;
        else is_point = ch == '.';
        if (is_digit) n = n * 10.0 + (ch - '0');
        ch = *++str;
    }
	return n * f;
}

inline void toQubit(char*& str, char*& n)
{
    eatWS(str);
    if (*str != 'q') 
        LOGERROR("expected q not %c", *str);
    str++;
    if (*str != '[') 
        LOGERROR("expected [ not %c", *str);
    str++;
    if (!isDigit(*str)) 
        LOGERROR("expected a digit but %c is found", *str);
    while (isDigit(*str)) {
        *n++ = *str++;
    }
    if (*str != ']')
        LOGERROR("expected ] not %c", *str);
    str++;
}


inline bool	match(const char* in, const int size, const char* ref) {
    if (*ref == '\0') return false;
    int c = 0;
    while (ref[c]) {
        if (ref[c] != in[c])
            return false;
        c++;
    }
    return size == c;
}

inline bool canAccess(const char* path, struct stat& st)
{
    if (stat(path, &st)) return false;
#ifdef _WIN32
#define R_OK 4
    if (_access(path, R_OK)) return false;
#else
    if (access(path, R_OK)) return false;
#endif
    return true;
}



struct Circuit {

    #define MAX_GATENAME_LEN 16
    #define MAX_QUBIT_DIGITS 32
    #define MAX_GATES 12

    const char* GATE_QASM[MAX_GATES] = {
        "i",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "cx",
        "cy",
        "cz",
        "swap",
        "iswap"
    };
    const char* GATE_STIM[MAX_GATES] = {
        "I",
        "X",
        "Y",
        "Z",
        "H",
        "S",
        "S_DAG",
        "CX",
        "CY",
        "CZ",
        "SWAP",
        "ISWAP"
    };

#if defined(__linux__) || defined(__CYGWIN__)
    int file;
#else
    ifstream file;
#endif
    Timer timer;
    string path;
    char* max_qubits;
    char* prev;
    char* qasm;
    char* stim;
    char* eof;
    size_t size;

    Circuit() :
        max_qubits(nullptr)
        , prev(nullptr)
        , qasm(nullptr)
        , stim(nullptr)
        , eof(nullptr)
        , size(0)
    { }

    ~Circuit() {
        if (max_qubits != nullptr)
            std::free(max_qubits);
        if (prev != nullptr)
            std::free(prev);
        if (stim != nullptr)
            std::free(stim);
        if (qasm != nullptr) {
#if defined(__linux__) || defined(__CYGWIN__)
            if (munmap(qasm, size) != 0)
                LOGERROR("cannot clean file mapping.");

#else
            std::free(qasm);
#endif
        }
        eof = nullptr;
        size = 0;
    }

    inline int translate_gate(char* in) {
        const int gatelen = strlen(in);
        for (int i = 0; i < MAX_GATES; i++) {
            const char* ref = GATE_QASM[i];
            int c = 0;
            while (ref[c]) {
                if (ref[c] != in[c])
                    break;     
                c++;
            }
            if (gatelen == c)
                return i;
        }
        LOGERROR("unknown gate %s.", in);
    }

    void read_qasm(const char* path) {
        if (path == nullptr)
            LOGERROR("circuit path is empty.");
        struct stat st;
        if (!canAccess(path, st))
            LOGERROR("circuit file is inaccessible.");
        size = st.st_size;
        LOG("Parsing circuit file \"%s\" (size: %zd MB)..", path, ratio(size, MB));
        timer.start();
        qasm = nullptr;
#if defined(__linux__) || defined(__CYGWIN__)
        file = open(path, O_RDONLY, 0);
        if (file == -1) LOGERROR("cannot open input file");
        qasm = static_cast<char*>(mmap(NULL, size, PROT_READ, MAP_PRIVATE, file, 0));
        close(file);
#else
        file.open(path, ifstream::in);
        if (!file.is_open()) LOGERROR("cannot open input file.");
        qasm = (char*)calloc(size + 1, sizeof(char));
        file.read(qasm, size);
        qasm[size] = '\0';
        file.close();
#endif
        eof = qasm + size;
        max_qubits = (char*) calloc(MAX_QUBIT_DIGITS + 1, sizeof(char));
        prev =  (char*) calloc(MAX_GATENAME_LEN + 1, sizeof(char));
        stim =  (char*) calloc(size + 1, sizeof(char));
        *prev = '\0';
        this->path = path;
        timer.stop();
        LOG(" done in %.2f milliseconds.\n", timer.time());
    }

    void read_gate(char*& from, char*& to) {       
        eatWS(from);
        char gate_qasm[MAX_GATENAME_LEN];
        int k = 0;
        int gatename_len = 0;
        while ((isalpha(from[gatename_len]) || from[gatename_len] == '_') && gatename_len < MAX_GATENAME_LEN) {
            gate_qasm[gatename_len] = from[gatename_len];
            gatename_len++;
        }
        if (gatename_len == MAX_GATENAME_LEN)
            LOGERROR("gate name is too long.");
        gate_qasm[gatename_len] = '\0';
        from += gatename_len;
        int expected_size = 1;
        int stim_gate_idx = translate_gate(gate_qasm);
        assert(stim_gate_idx < MAX_GATES);
        const char* gate_stim = GATE_STIM[stim_gate_idx];
        gatename_len = strlen(gate_stim);
        if (match(gate_stim, gatename_len, prev))
            *to++ = ' ';
        else {
            if (*prev != '\0') {
                #if defined(__linux__) || defined(__CYGWIN__)
                *to++ = '\r';
                #endif
                *to++ = '\n';
            }
            k = 0; 
            while (k < gatename_len)
                *to++ = gate_stim[k++];
            *to++ = ' ';
        }
        // Read gate inputs   
        unsigned sign = 0;
        while (*from != ';' && from < eof) {
            if (*from++ == ',') *to++ = ' ';
            toQubit(from, to);
            if (expected_size == 2) {
                toQubit(from, to);
            }
        }
        from++; // skip (;)
        k = 0;
        char* p = prev;
        while (k < gatename_len)
            *p++ = gate_stim[k++];
        *p++ = '\0';
    }

    void to_stim() {
        LOG(" Translating QASM circuit to Stim..");
        timer.start();
        char* from = qasm;
        char* to = stim;
        while (from < eof) {
            eatWS(from);
            if (*from == '\0') break;
            if (match(from, 8, "OPENQASM")) {
                from += 8;
                double version = toFloat(from);
                if (version != 2.0)
                    LOGERROR("QASM version %.3f not compatible.", version);
                eatLine(from);
            }
            else if (match(from, 4, "qreg")) {
                from += 4;
                char* qubits = max_qubits;
                toQubit(from, qubits);
                *qubits = '\0';
                qubits = max_qubits;
                *to++ = '#';
                while (*qubits)
                    *to++ = *qubits++;
                #if defined(__linux__) || defined(__CYGWIN__)
                *to++ = '\r';
                #endif
                *to++ = '\n';
                eatLine(from);
            }
            else if (match(from, 7, "include")) {
                eatLine(from);
            }
            else if (match(from, 4, "gate")) {
                eatLine(from);
            }
            else {      
                read_gate(from, to);
            }
        }
        #if defined(__linux__) || defined(__CYGWIN__)
        *to++ = '\r';
        #endif
        *to++ = '\n';
        timer.stop();
        LOG("(found %s qubits) done in %.2f milliseconds.\n", max_qubits, timer.time());
        size_t lastidx = path.find_last_of(".");
        string stim_file_path = path.substr(0, lastidx) + ".stim";
        LOG(" Writting Stim circuit to file %s..", stim_file_path.c_str());
        timer.start();
        FILE* stim_file = nullptr;
        if (stim_file == nullptr) {
            if ((stim_file = fopen(stim_file_path.c_str(), "w")) == nullptr)
                LOGERROR("Stim file path does not exist.");
        }
        size_t stim_size = to - stim;
        fwrite(stim, 1, stim_size, stim_file);
		if (stim_file != nullptr) {
			fclose(stim_file);
			stim_file = nullptr;
		}
        from = nullptr;
        to = nullptr;
        timer.stop();
        LOG(" done in %.2f milliseconds.\n", timer.time());
    }

};


int main(int argc, char** argv) {
    Timer timer;
    Circuit* circuit = nullptr;

    if (argc == 1) LOGERROR("path to qasm directory is missing.");

    std::string path = argv[1];
    for (const auto & entry : fs::directory_iterator(path)) {
        string file_path{ entry.path() };
        if (entry.path().extension() == ".qasm") {
            struct stat st;
            if (!canAccess(file_path.c_str(), st))
                LOGERROR("file path %s is inaccessible.", file_path.c_str());
            circuit = new Circuit();
            circuit->read_qasm(file_path.c_str());
            circuit->to_stim();
            delete circuit;
            circuit = nullptr;
            LOG("\n");
        }
    }

    return 0;
}

