
#ifndef __GL_MACROS_
#define __GL_MACROS_

#include "datatypes.hpp"
#include <cstring>

namespace QuaSARQ {

	constexpr size_t KB = 0x00000400;
	constexpr size_t MB = 0x00100000;
	constexpr size_t GB = 0x40000000;

	#define UNDEFINED -1
	#define MIN(x,y)		((x) < (y) ? (x) : (y))
	#define MAX(x,y)		((x) > (y) ? (x) : (y))
	#define RESETSTRUCT(MEMPTR) std::memset(MEMPTR, 0, sizeof(*MEMPTR));

	constexpr double ratio		(const double& x, const double& y) { return y ? x / y : 0; }
	constexpr size_t ratio		(const size_t& x, const size_t& y) { return y ? x / y : 0; }
	constexpr double percent	(const double& x, const double& y) { return ratio(100 * x, y); }
	
}

#endif