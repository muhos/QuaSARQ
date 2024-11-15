
#ifndef __CU_MEMORY_
#define __CU_MEMORY_

#define cuMalloc cudaMallocAsync

#include <map>

#include "definitions.cuh"
#include "vector.cuh"
#include "constants.hpp"
#include "logging.hpp"
#include "malloc.hpp"

namespace QuaSARQ {

	typedef void*			addr_t;
	typedef std::map		< addr_t, size_t > alloc_list_t;
	typedef std::multimap	< size_t, addr_t > free_list_t;

	struct Pool {
		addr_t mem;
		size_t cap;
		size_t off;
		Pool() : mem(nullptr), cap(0), off(0) {}
	};

	struct GPU_memory_exception {
		GPU_memory_exception() { fflush(stderr); fflush(stdout); }
	};

	class DeviceAllocator {

		static constexpr size_t ALIGNMENT = 8;
		static constexpr size_t GPU_PENALTY = 256 * MB;
		static constexpr size_t CPU_PENALTY = 512 * MB;

		// CPU pool
		Pool _cpool;

		// GPU pool
		Pool _gpool;

		// Memory availability
		size_t	_ctot, _cfree;
		size_t	_gtot, _gfree;
		size_t	_climit, _glimit;


		// Trackers
		free_list_t  cpu_free_list, gpu_free_list;
		alloc_list_t cpu_alloc_list, gpu_alloc_list;

		size_t align_down(const size_t& size, size_t alignment = ALIGNMENT) { return size & ~(alignment - 1); }
		size_t align_up(const size_t& size, size_t alignment = ALIGNMENT) { return (size + (alignment - 1)) & ~(alignment - 1); }
		bool is_aligned(const size_t& size, size_t alignment = ALIGNMENT) { return size == align_down(size, alignment); }

	public:

		DeviceAllocator();
		DeviceAllocator(const size_t& cpu_limit, const size_t& gpu_limit);
		
		bool destroy_gpu_pool();
		bool destroy_cpu_pool();
		bool create_gpu_pool(const size_t& limit = 0);
		bool create_cpu_pool(const size_t& limit = 0);
		bool resize_gpu_pool(const size_t& new_size);
		bool resize_cpu_pool(const size_t& new_size);

		template <class T>
		T* allocate(const size_t& new_size) {
			if (!new_size)
				return nullptr;
			const size_t bytes = new_size * sizeof(T);
			T* object_ptr = nullptr;
			size_t aligned_size = align_up(bytes);
			free_list_t::iterator free_block = gpu_free_list.lower_bound(aligned_size);
			// found free block.
			if (free_block != gpu_free_list.end()) {
				assert(free_block->second != nullptr);
				aligned_size = free_block->first;
				assert(is_aligned(aligned_size));
				gpu_free_list.erase(free_block);	
				gpu_alloc_list.insert(std::make_pair(free_block->second, aligned_size));
				object_ptr = static_cast<T*>(free_block->second);
			}
			// no free blocks, allocate new one from the pool.
			else {
				if (!_gpool.cap || aligned_size > _gpool.cap) {
					LOGERRORN("No memory space left for allocator.");
					throw GPU_memory_exception();
				}
				assert((_glimit && _gpool.off <= _glimit) || (!_glimit && _gpool.off <= _gfree));
				byte_t* start_ptr = static_cast<byte_t*>(_gpool.mem);
				object_ptr = reinterpret_cast<T*>(start_ptr + _gpool.off);
				_gpool.off += aligned_size;
				assert(_gpool.cap >= aligned_size);
				_gpool.cap -= aligned_size;				
				gpu_alloc_list.insert(std::make_pair(static_cast<addr_t>(object_ptr), aligned_size));
			}
			assert(object_ptr != nullptr);
			return object_ptr;
		}

		template <class T>
		void deallocate(T* ptr) {
			addr_t void_ptr = static_cast<addr_t>(ptr);
			alloc_list_t::iterator allocated_block = gpu_alloc_list.find(void_ptr);
			if (allocated_block == gpu_alloc_list.end()) {
				LOGERRORN("memory block %p is not allocated via the GPU allocator", void_ptr);
				throw GPU_memory_exception();
			}
			const size_t size = allocated_block->second;
			assert(is_aligned(size));
			gpu_alloc_list.erase(allocated_block);
			gpu_free_list.insert(std::make_pair(size, void_ptr));
		}

		template <class T>
		T* allocate_pinned(const size_t& new_size) {
			if (!new_size)
				return nullptr;
			const size_t bytes = new_size * sizeof(T);
			T* object_ptr = nullptr;
			size_t aligned_size = align_up(bytes);
			free_list_t::iterator free_block = cpu_free_list.lower_bound(aligned_size);
			// found free block.
			if (free_block != cpu_free_list.end()) {
				assert(free_block->second != nullptr);
				aligned_size = free_block->first;
				assert(is_aligned(aligned_size));
				cpu_free_list.erase(free_block);
				cpu_alloc_list.insert(std::make_pair(free_block->second, aligned_size));
				object_ptr = static_cast<T*>(free_block->second);
			}
			// no free blocks, allocate new one from the pool.
			else {
				if (!_cpool.cap || aligned_size > _cpool.cap) {
					LOGERRORN("No memory space left for allocator.");
					throw CPU_memory_exception();
				}
				assert((_climit && _cpool.off <= _climit) || (!_climit && _cpool.off <= _cfree));
				byte_t* start_ptr = static_cast<byte_t*>(_cpool.mem);
				object_ptr = reinterpret_cast<T*>(start_ptr + _cpool.off);
				_cpool.off += aligned_size;
				assert(_cpool.cap >= aligned_size);
				_cpool.cap -= aligned_size;
				cpu_alloc_list.insert(std::make_pair(static_cast<addr_t>(object_ptr), aligned_size));
			}
			assert(object_ptr != nullptr);
			return object_ptr;
		}

		template <class T>
		void deallocate_pinned(T* ptr) {
			addr_t void_ptr = static_cast<addr_t>(ptr);
			alloc_list_t::iterator allocated_block = cpu_alloc_list.find(void_ptr);
			if (allocated_block == cpu_alloc_list.end()) {
				LOGERRORN("memory block %p is not allocated via the CPU allocator", void_ptr);
				throw CPU_memory_exception();
			}
			const size_t size = allocated_block->second;
			assert(is_aligned(size));
			cpu_alloc_list.erase(allocated_block);
			cpu_free_list.insert(std::make_pair(size, void_ptr));
		}

		template <class T>
		void resize(T*& ptr, const size_t& new_size) {
			if (ptr != nullptr) {
				addr_t void_ptr = static_cast<addr_t>(ptr);
				alloc_list_t::iterator allocated_block = gpu_alloc_list.find(void_ptr);
				if (allocated_block == gpu_alloc_list.end()) {
					LOGERRORN("memory block %p is not allocated via the GPU allocator", void_ptr);
					throw GPU_memory_exception();
				}
				const size_t size = allocated_block->second;
				assert(is_aligned(size));
				if (size >= new_size)
					return;
				deallocate<T>(ptr);
				ptr = nullptr;
			}
			assert(ptr == nullptr);
			ptr = allocate<T>(new_size);
			assert(ptr != nullptr);
		}

		template <class T>
		void resize_pinned(T*& ptr, const size_t& new_size) {
			if (ptr != nullptr) {
				addr_t void_ptr = static_cast<addr_t>(ptr);
				alloc_list_t::iterator allocated_block = cpu_alloc_list.find(void_ptr);
				if (allocated_block == cpu_alloc_list.end()) {
					LOGERRORN("memory block %p is not allocated via the CPU allocator", void_ptr);
					throw CPU_memory_exception();
				}
				const size_t size = allocated_block->second;
				assert(is_aligned(size));
				if (size >= new_size)
					return;
				deallocate_pinned<T>(ptr);
				ptr = nullptr;
			}
			assert(ptr == nullptr);
			ptr = allocate_pinned<T>(new_size);
			assert(ptr != nullptr);
		}

		template <class T>
		T* resize_vector(cuVec<T>*& vector, const size_t& new_size) {
			bool first = vector == nullptr;
			if (first) vector = allocate<cuVec<T>>(1);
			assert(vector != nullptr);
			cuVec<T> tmp;
			T* vector_data = nullptr;
			if (!first) {
				CHECK(cudaMemcpy(&tmp, vector, sizeof(cuVec<T>), cudaMemcpyDeviceToHost));
				vector_data = tmp.data();
				assert(vector_data != nullptr);
				if (tmp.size() >= new_size)
					return vector_data;
				else 
					deallocate<T>(vector_data);
			}
			assert(vector_data == nullptr);
			vector_data = allocate<T>(new_size);
			assert(vector_data != nullptr);
			tmp.alloc(vector_data, new_size);
			tmp.resize(new_size);
			CHECK(cudaMemcpy(vector, &tmp, sizeof(cuVec<T>), cudaMemcpyHostToDevice));
			assert(vector_data != nullptr);
			return vector_data;
		}

		size_t cpu_capacity() const { return _cpool.cap; }
		size_t gpu_capacity() const { return _gpool.cap; }
		
	};

}

#endif