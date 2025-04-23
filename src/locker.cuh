#pragma once

#include "definitions.cuh"
#include "memory.cuh"

namespace QuaSARQ {


	class DeviceLocker {

		int* mutex;

	public:

		DeviceLocker(): mutex(nullptr) {}
		~DeviceLocker() { mutex = nullptr; }

		void alloc(int* mutex) {
			this->mutex = mutex;
		}

		NOINLINE_DEVICE void lock();

		NOINLINE_DEVICE bool unlocked();

		NOINLINE_DEVICE void unlock();

	};

	class Locker {

		DeviceAllocator& allocator;

		DeviceLocker* _dlocker;
		int* _mutex;

	public:

		Locker(DeviceAllocator& allocator) : 
            	allocator(allocator) 
			, 	_dlocker(nullptr)
			,	_mutex(nullptr)
			{}

		void alloc() {
			_dlocker = allocator.allocate<DeviceLocker>(1);
			_mutex = allocator.allocate<int>(1);

			DeviceLocker *tmp = new DeviceLocker();
			assert(_mutex != nullptr);
			tmp->alloc(_mutex);

            CHECK(cudaMemcpy(_dlocker, tmp, sizeof(DeviceLocker), cudaMemcpyHostToDevice));

			delete tmp;
		}

		void reset(const cudaStream_t& s) {
			CHECK(cudaMemsetAsync(_mutex, 0, sizeof(int), s));
		}

		DeviceLocker* deviceLocker() { return _dlocker; }

	};

}