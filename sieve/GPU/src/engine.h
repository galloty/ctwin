/*
Copyright 2024, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"

class engine : public device
{
private:
	cl_mem _prime_vector = nullptr, _ak_a2k_vector = nullptr, _factor_vector = nullptr;
	cl_mem _prime_count = nullptr, _factor_count = nullptr;
	cl_kernel _check_primes = nullptr, _init_factors = nullptr, _check_factors = nullptr, _clear_primes = nullptr;

public:
	engine(const platform & platform, const size_t d) : device(platform, d, true) {}
	virtual ~engine() {}

public:
	void allocMemory(const size_t prime_size, const size_t factor_size)
	{
#if defined (ocl_debug)
		std::cerr << "Alloc gpu memory." << std::endl;
#endif
		_prime_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong3) * prime_size);	// p, q, one
		_ak_a2k_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);
		_factor_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * factor_size);
		_prime_count = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
		_factor_count = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
	}

public:
	void releaseMemory()
	{
#if defined (ocl_debug)
		std::cerr << "Free gpu memory." << std::endl;
#endif
		_releaseBuffer(_prime_vector);
		_releaseBuffer(_ak_a2k_vector);
		_releaseBuffer(_factor_vector);
		_releaseBuffer(_prime_count);
		_releaseBuffer(_factor_count);
	}

public:
	void createKernels()
	{
#if defined (ocl_debug)
		std::cerr << "Create ocl kernels." << std::endl;
#endif
		_check_primes = _createKernel("check_primes");
		_setKernelArg(_check_primes, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_check_primes, 1, sizeof(cl_mem), &_prime_vector);

		_init_factors = _createKernel("init_factors");
		_setKernelArg(_init_factors, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_init_factors, 1, sizeof(cl_mem), &_prime_vector);
		_setKernelArg(_init_factors, 2, sizeof(cl_mem), &_ak_a2k_vector);

		_check_factors = _createKernel("check_factors");
		_setKernelArg(_check_factors, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_check_factors, 1, sizeof(cl_mem), &_prime_vector);
		_setKernelArg(_check_factors, 2, sizeof(cl_mem), &_ak_a2k_vector);
		_setKernelArg(_check_factors, 3, sizeof(cl_mem), &_factor_count);
		_setKernelArg(_check_factors, 4, sizeof(cl_mem), &_factor_vector);

		_clear_primes = _createKernel("clear_primes");
		_setKernelArg(_clear_primes, 0, sizeof(cl_mem), &_prime_count);
	}

public:
	void releaseKernels()
	{
#if defined (ocl_debug)
		std::cerr << "Release ocl kernels." << std::endl;
#endif
		_releaseKernel(_check_primes);
		_releaseKernel(_init_factors);
		_releaseKernel(_check_factors);
		_releaseKernel(_clear_primes);
	}

public:
	void clearPrimeCount() { const cl_uint zero = 0; _writeBuffer(_prime_count, &zero, sizeof(cl_uint)); }
	cl_uint readPrimeCount() { cl_uint count; _readBuffer(_prime_count, &count, sizeof(cl_uint)); return count; }
	void clearFactorCount() { const cl_uint zero = 0; _writeBuffer(_factor_count, &zero, sizeof(cl_uint)); }
	cl_uint readFactorCount() { cl_uint count; _readBuffer(_factor_count, &count, sizeof(cl_uint)); return count; }
	void readFactors(cl_ulong2 * const ptr, const size_t count) { if (count > 0) _readBuffer(_factor_vector, ptr, sizeof(cl_ulong2) * count); }

public:
	void checkPrimes(const size_t count, const uint64_t index)
	{
		const cl_ulong i = cl_ulong(index);
		_setKernelArg(_check_primes, 2, sizeof(cl_ulong), &i);
		_executeKernel(_check_primes, count);
	}

public:
	void initFactors(const size_t count)
	{
		_executeKernel(_init_factors, count);
	}

public:
	void checkFactors(const size_t count)
	{
		_executeKernel(_check_factors, count);
	}

public:
	void clearPrimes()
	{
		_executeKernel(_clear_primes, 1);
	}
};
