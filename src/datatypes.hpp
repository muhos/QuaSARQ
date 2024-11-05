
#ifndef __DATATYPES_H
#define __DATATYPES_H

#include <cstddef>
#include <cstdint>

namespace QuaSARQ {

	// primitive types
	typedef const char* arg_t;
	typedef unsigned char byte_t;
	typedef unsigned short uint16;
	typedef unsigned int uint32;
	typedef signed long long int int64;
	typedef unsigned long long int uint64;

	// Tableau based
	enum InitialState {
		Zero,
		Plus, 
		Imag
	};
	
}

#endif