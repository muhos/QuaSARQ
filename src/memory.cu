
#include "memory.cuh"
#include "malloc.hpp"

using namespace QuaSARQ;

DeviceAllocator::DeviceAllocator() :
	  _cpool()
	, _gpool()
	, _ctot(0)
	, _cfree(0)
	, _gtot(0)
	, _gfree(0)
	, _climit(0)
	, _glimit(0)
{ }

DeviceAllocator::DeviceAllocator(const size_t& cpu_limit, const size_t& gpu_limit) :
	  _cpool()
	, _gpool()
	, _ctot(0)
	, _cfree(0)
	, _gtot(0)
	, _gfree(0)
	, _climit(0)
	, _glimit(0)
{ 
	if (!create_cpu_pool(cpu_limit))
		throw CPU_memory_exception();
	if (!create_gpu_pool(gpu_limit))
		throw GPU_memory_exception();
}

bool DeviceAllocator::destroy_gpu_pool() {
	if (_gpool.mem == nullptr) 
		return true;
	// Nullify dangling pointers
	LOGN2(1, "Freeing GPU memory pool.. ");
	for (free_list_t::iterator i = gpu_free_list.begin(); i != gpu_free_list.end(); i++) {
		i->second = nullptr;
	}
	gpu_free_list.clear();
	gpu_alloc_list.clear();
	// Free memory
	if (_gpool.mem != nullptr) {
		assert(_gpool.cap);
		if (cudaFree(_gpool.mem) != cudaSuccess)
			return false;
		_gpool.mem = nullptr;
		_gpool.off = _gpool.cap = 0;
	}
	LOGDONE(1, 3);
	return true;
}

bool DeviceAllocator::destroy_cpu_pool() {
	if (_cpool.mem == nullptr)
		return true;
	// Nullify dangling pointers
	LOGN2(1, "Freeing CPU memory pool.. ");
	for (free_list_t::iterator i = cpu_free_list.begin(); i != cpu_free_list.end(); i++) {
		i->second = nullptr;
	}
	cpu_free_list.clear();
	cpu_alloc_list.clear();
	// Free memory
	if (_cpool.mem != nullptr) {
		assert(_cpool.cap);
		if (cudaFreeHost(_cpool.mem) != cudaSuccess)
			return false;
		_cpool.mem = nullptr;
		_cpool.off = _cpool.cap = 0;
	}
	LOGDONE(1, 3);
	return true;
}

bool DeviceAllocator::create_gpu_pool(const size_t& limit) {
	if (_gpool.mem != nullptr && _gpool.cap > 0) {
		LOG2(1, "GPU memory pool is already created.");
		return true;
	}
	assert(_gpool.mem == nullptr);
	_glimit = align_up(limit);
	cudaMemGetInfo(&_gfree, &_gtot);
	// Assume that CPU free memory is at least
	// as much as the GPU memory.
	_cfree = _gfree;
	if (_gfree > GPU_PENALTY)
		_gfree -= GPU_PENALTY;
	if (_glimit > _gfree) {
		LOGWARNING("Requested %zd MB but %zd MB is only free.", ratio(_glimit, MB), ratio(_gfree, MB));
		LOGWARNING("Only %zd MB will be allocated.", ratio(_gfree, MB));
		_gpool.cap = _gfree;
	}
	else if (_glimit) {
		is_aligned(_glimit);
		_gpool.cap = _glimit;
	}
	else
		_gpool.cap = _gfree;
	assert(_gpool.off == 0);
	if (cuMalloc((addr_t*)& _gpool.mem, _gpool.cap, 0) != cudaSuccess)
		return false;
	LOG2(1, "GPU memory pool of %zd MB is created successfully.", ratio(_gpool.cap, MB));
	LOGN2(1, "Initializing GPU memory pool to 0.. ");
	CHECK(cudaMemsetAsync(_gpool.mem, 0, _gpool.cap));
    LOGDONE(1, 3);
	return true;
}

bool DeviceAllocator::resize_gpu_pool(const size_t& new_size) {
	if (_gpool.mem == nullptr) {
		LOGWARNING("cannot resize an empty memory pool.");
		return false;
	}
	// Nullify dangling pointers
	LOGN2(1, "Resizing GPU memory pool to %zd MB.. ", ratio(new_size, MB));
	for (free_list_t::iterator i = gpu_free_list.begin(); i != gpu_free_list.end(); i++) {
		i->second = nullptr;
	}
	gpu_free_list.clear();
	gpu_alloc_list.clear();
	// Free memory
	if (_gpool.mem != nullptr) {
		assert(_gpool.cap);
		if (cudaFree(_gpool.mem) != cudaSuccess)
			return false;
		_gpool.mem = nullptr;
		_gpool.off = _gpool.cap = 0;
	}
	// Allocate new memory
	if (new_size > _gfree) {
		LOG2(1, "failed due to insufficient memory.");
	}
	_gpool.cap = new_size;
	_glimit = new_size;
	assert(_gpool.off == 0);
	if (cuMalloc((addr_t*)& _gpool.mem, _gpool.cap, 0) != cudaSuccess)
		return false;
	LOG2(1, "succeeded");
	LOGN2(1, "Initializing GPU memory pool to 0.. ");
	CHECK(cudaMemsetAsync(_gpool.mem, 0, _gpool.cap));
    LOGDONE(1, 3);
	return true;
}

bool DeviceAllocator::create_cpu_pool(const size_t& limit) {
	if (_cpool.mem != nullptr && _cpool.cap > 0) {
		LOG2(1, "CPU memory pool is already created.");
		return true;
	}
	assert(_cpool.mem == nullptr);
	_climit = align_up(limit);
	if (!_cfree) cudaMemGetInfo(&_cfree, &_ctot);
	if (_cfree > CPU_PENALTY)
		_cfree -= CPU_PENALTY;
	if (_climit > _cfree) {
		LOGWARNING("Requested %zd MB but %zd MB is only free.", ratio(_climit, MB), ratio(_cfree, MB));
		LOGWARNING("Only %zd MB will be allocated.", ratio(_cfree, MB));
		_cpool.cap = _cfree;
	}
	else if (_climit) {
		is_aligned(_climit);
		_cpool.cap = _climit;
	}
	else
		_cpool.cap = _cfree;
	assert(_cpool.off == 0);
	if (cudaMallocHost((addr_t*)&_cpool.mem, _cpool.cap) != cudaSuccess)
		return false;
	LOG2(1, "CPU memory pool of %zd KB is created successfully.", ratio(_cpool.cap, KB));
	return true;
}

bool DeviceAllocator::resize_cpu_pool(const size_t& new_size) {
	if (_cpool.mem == nullptr) {
		LOGERRORN("cannot resize an empty memory pool.");
		return false;
	}
	// Nullify dangling pointers
	LOGN2(1, "Resizing CPU memory pool to %zd KB.. ", ratio(new_size, KB));
	for (free_list_t::iterator i = cpu_free_list.begin(); i != cpu_free_list.end(); i++) {
		i->second = nullptr;
	}
	cpu_free_list.clear();
	cpu_alloc_list.clear();
	// Free memory
	if (_cpool.mem != nullptr) {
		assert(_cpool.cap);
		if (cudaFreeHost(_cpool.mem) != cudaSuccess)
			return false;
		_cpool.mem = nullptr;
		_cpool.off = _cpool.cap = 0;
	}
	// Allocate new memory
	if (new_size > _gfree) {
		LOG2(1, "failed due to insufficient memory.");
	}
	_cpool.cap = new_size;
	_climit = new_size;
	assert(_cpool.off == 0);
	if (cudaMallocHost((addr_t*)& _cpool.mem, _cpool.cap) != cudaSuccess)
		return false;
	LOG2(1, "succeeded");
	return true;
}