#pragma once

#include <cstddef>
#include <cstdint>

namespace QuaSARQ {

	// primitive types
	typedef const char* arg_t;
	typedef unsigned char uint8;
	typedef uint8 byte_t;
	typedef unsigned short uint16;
	typedef unsigned int uint32;
	typedef uint32 qubit_t;
	typedef uint32 depth_t;
	typedef signed long long int int64;
	typedef unsigned long long int uint64;

	// Tableau based
	enum InitialState {
		Zero,
		Plus, 
		Imag
	};

	// Circuit generation modes
	enum CircuitMode { 
		RANDOM_CIRCUIT, 
		PARSED_CIRCUIT 
	};

	// Memory contexts.
	enum Context { CPU, GPU, UNKNOWN };
	
}