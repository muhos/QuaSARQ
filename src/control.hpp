#pragma once

#include "definitions.cuh"
#include "definitions.hpp"
#include "datatypes.hpp"
#include "options.hpp"

namespace QuaSARQ {

	size_t	sysMemUsed				();
	size_t	getAvailSysMem			();
	void	getBuildInfo			(const int& verbose);
	void	getCPUInfo				(const int& verbose);
	int		getGPUInfo				(const int& verbose);
	void	signal_termination		(void h_intr(int));
	void	signal_timeout			(void h_timeout(int));
	void	set_timeout				(int);
	void	set_memoryout			(int);
	void	handler_terminate		(int);
	void	segmentation_fault		(int);
	void	illegal_code			(int);
	void	arithmetic_error		(int);

	#define EXIT_INTERRUPTED 2
	#define FAULT_DETECTOR \
	{ \
		signal(SIGSEGV, segmentation_fault); \
		signal(SIGILL, illegal_code); \
		signal(SIGFPE, arithmetic_error); \
	}
}