
#include <string>
#include <cuda_runtime.h>
#ifdef  _WIN32
#include <Windows.h>
#endif

#include "version.hpp"

namespace QuaSARQ {

	#define BUFFERSIZE 256
	char buffer[BUFFERSIZE];

	#ifndef VERSION
		#define VERSION "1.0"
	#endif

	#ifndef COMPILER
		#if defined (_MSC_VER)
			#define COMPILER "Visual C++"
		#elif defined(__GNUC__)
			#ifdef __VERSION__
				#define COMPILER "g++ v" __VERSION__
			#else
				#define COMPILER "g++"
			#endif
		#else
			#define COMPILER "unknown"
		#endif
	#endif

	#ifndef OSYSTEM
		#if defined(__linux__) || defined(__CYGWIN__ )
			#define OSYSTEM "Linux"
		#else
			#define OSYSTEM "unknown"
		#endif
	#endif

	#ifndef DATE
		#define DATE __DATE__ " " __TIME__
	#endif

	const char* version() { return "v" VERSION; }

	const char* signature() { return "quasarq-" VERSION; }

	const char* compiler() 
	{
#ifdef _MSC_VER
		std::string FULLSTR(COMPILER);
		FULLSTR += " version " + std::to_string(_MSC_VER);
#ifdef CUDA_VERSION
		FULLSTR += " + NVCC version " + std::to_string(CUDA_VERSION);
#endif
		if (FULLSTR.length() <= BUFFERSIZE) {
			strcpy(buffer, FULLSTR.c_str());
			return buffer;
		}
		else 
			return COMPILER;
#else 
		return COMPILER;
#endif
	}

	const char* compilemode()
	{
#if defined(_DEBUG) || defined(DEBUG)
		return "(debug mode)";
#elif defined(NDEBUG)
		return "(release mode)";
#else 
		return "(asserting mode)";
#endif
	}

#if defined(_WIN32)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

	const char* osystem() 
	{ 
#ifdef _WIN32
		OSVERSIONINFOEX info;
		ZeroMemory(&info, sizeof(OSVERSIONINFOEX));
		info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
		GetVersionEx((LPOSVERSIONINFO)&info);
		int type = (WINVER >> 8);
		std::string FULLSTR("Windows ");
		FULLSTR += std::to_string(type) + " ";
		FULLSTR += "version " + std::to_string(info.dwMajorVersion) + "." +  std::to_string(info.dwMinorVersion) + " ";
		FULLSTR += "build " + std::to_string(info.dwBuildNumber);
		if (FULLSTR.length() <= BUFFERSIZE) {
			strcpy(buffer, FULLSTR.c_str());
			return buffer;
		}
		else
			return OSYSTEM;
#else
		return OSYSTEM; 
#endif
	}
	const char* date() { return DATE; }

#if defined(_WIN32)
#pragma warning(pop)
#endif

}