
#include "locker.cuh"

namespace QuaSARQ {

    NOINLINE_DEVICE void DeviceLocker::lock() {
        assert(mutex != nullptr);
        while (atomicCAS(mutex, 0, 1) != 0);
    }

    NOINLINE_DEVICE void DeviceLocker::unlock() {
        assert(mutex != nullptr);
        atomicExch(mutex, 0);
    }

}