#pragma once

#include <iostream>
#include <algorithm>
#include <cstring>
#include <locale>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <climits>
#include <cstdlib>
#include <csignal>
#include <cstdint>
#include <sys/stat.h>
#include "datatypes.hpp"
#include "constants.hpp"
#include "logging.hpp"

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <fcntl.h>
#include <unistd.h>
#include <cpuid.h>
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

namespace QuaSARQ {


	inline bool isDigit				(const char& ch) { return (ch ^ 48) <= 9; }
	inline bool isSpace				(const char& ch) { return (ch >= 9 && ch <= 13) || ch == 32; }
	inline void eatWS				(char*& str) { while (isSpace(*str)) str++; }
	inline void eatLine				(char*& str) { while (*str) if (*str++ == '\n') return; }
	inline uint32 toInteger			(char*& str)
	{
		eatWS(str);
		if (!isDigit(*str)) LOGERROR("expected a digit but 0x%0X is found", *str);
		uint32 n = 0;
		while (isDigit(*str)) n = n * 10 + (*str++ - '0');
		return n;
	}
	inline size_t toInteger			(char*& str, uint32& sign)
	{
		eatWS(str);
		sign = 0;
		if (*str == '-') sign = 1, str++;
		else if (*str == '+') str++;
		if (!isDigit(*str)) LOGERROR("expected a digit but %c is found", *str);
		size_t n = 0;
		while (isDigit(*str)) n = n * 10ULL + (*str++ - '0');
		return n;
	}
	inline uint32 nextPow2(uint32 x) {
		if (x <= 1) {
			return 1;
		}
		x--;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		x++;
		return x;
	}
	template<class T>
	inline bool		eq				(T& in, const char* ref) {
		while (*ref) { if (*ref != *in) return false; ref++; in++; }
		return true;
	}
	template<class T>
	inline bool		eqn				(T in, const char* ref, const bool& lower = false) {
		if (lower) {
			while (*ref) { 
				if (tolower(*ref) != tolower(*in))
					return false; 
				ref++; in++;
			}
		}
		else {
			while (*ref) { if (*ref != *in) return false; ref++; in++; }
		}
		return true;
	}
	inline size_t	hasstr			(const char* in, const char* ref)
	{
		size_t count = 0;
		const size_t reflen = strlen(ref);
		while (*in) {
			if (ref[count] != *in)
				count = 0;
			else
				count++;
			in++;
			if (count == reflen)
				return count;
		}
		return 0;
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
	
}