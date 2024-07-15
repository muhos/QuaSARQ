
#ifndef __CONTROL_
#define __CONTROL_

#include "definitions.cuh"
#include "definitions.hpp"
#include "datatypes.hpp"

namespace QuaSARQ {

	size_t	sysMemUsed				();
	size_t	getAvailSysMem			();
	void	getBuildInfo			();
	void	getCPUInfo				();
	int		getGPUInfo				();
	void	signal_handler			(void h_intr(int), void h_timeout(int) = nullptr);
	void	set_timeout				(int);
	void	set_memoryout			(int);
	void	handler_terminate		(int);
	void	handler_mercy_interrupt	(int);
	void	handler_mercy_timeout	(int);
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

#endif 