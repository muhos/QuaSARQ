
#ifndef __CU_TUNE_H
#define __CU_TUNE_H

#include "tableau.cuh"
#include "datatypes.cuh"

namespace QuaSARQ {

	class Tuner : public Simulator {

		void reset();

	public:

		Tuner() : Simulator() {}

		Tuner(const string& path) : Simulator(path) {}
		
		void write();
		void run();

	};

}

#endif 