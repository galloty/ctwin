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
	cl_mem _kro_vector = nullptr, _prime_vector = nullptr, _ext_vector = nullptr, _ext2_vector = nullptr, _factor_vector = nullptr;
	cl_mem _prime_count = nullptr, _factor_count = nullptr;
	cl_kernel _generate_primes_pos = nullptr, _init_factors_pos = nullptr, _generate_factors_pos = nullptr;
	cl_kernel _generate_primes_neg = nullptr, _init_factors_neg = nullptr, _generate_factors_neg = nullptr;
	cl_kernel _clear_primes = nullptr;

public:
	engine(const platform & platform, const size_t d) : device(platform, d, true) {}
	virtual ~engine() {}

public:
	void allocMemory(const size_t prime_size, const size_t factor_size)
	{
#if defined (ocl_debug)
		std::cerr << "Alloc gpu memory." << std::endl;
#endif
		_kro_vector = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_char) * 128 * 256);
		_prime_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);	// p, q
		_ext_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong) * prime_size);		// one / a^k
		_ext2_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);	// (one, 0) / b_1, b_2
		_factor_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * factor_size);	// p, b
		_prime_count = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
		_factor_count = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
	}

public:
	void releaseMemory()
	{
#if defined (ocl_debug)
		std::cerr << "Free gpu memory." << std::endl;
#endif
		_releaseBuffer(_kro_vector);
		_releaseBuffer(_prime_vector);
		_releaseBuffer(_ext_vector);
		_releaseBuffer(_ext2_vector);
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
		_generate_primes_pos = _createKernel("generate_primes_pos");
		_setKernelArg(_generate_primes_pos, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_generate_primes_pos, 1, sizeof(cl_mem), &_prime_vector);
		_setKernelArg(_generate_primes_pos, 2, sizeof(cl_mem), &_ext_vector);

		_init_factors_pos = _createKernel("init_factors_pos");
		_setKernelArg(_init_factors_pos, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_init_factors_pos, 1, sizeof(cl_mem), &_prime_vector);
		_setKernelArg(_init_factors_pos, 2, sizeof(cl_mem), &_ext_vector);
		_setKernelArg(_init_factors_pos, 3, sizeof(cl_mem), &_kro_vector);

		_generate_factors_pos = _createKernel("generate_factors_pos");
		_setKernelArg(_generate_factors_pos, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_generate_factors_pos, 1, sizeof(cl_mem), &_prime_vector);
		_setKernelArg(_generate_factors_pos, 2, sizeof(cl_mem), &_ext_vector);
		_setKernelArg(_generate_factors_pos, 3, sizeof(cl_mem), &_factor_count);
		_setKernelArg(_generate_factors_pos, 4, sizeof(cl_mem), &_factor_vector);

		_generate_primes_neg = _createKernel("generate_primes_neg");
		_setKernelArg(_generate_primes_neg, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_generate_primes_neg, 1, sizeof(cl_mem), &_prime_vector);
		_setKernelArg(_generate_primes_neg, 2, sizeof(cl_mem), &_ext2_vector);

		_init_factors_neg = _createKernel("init_factors_neg");
		_setKernelArg(_init_factors_neg, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_init_factors_neg, 1, sizeof(cl_mem), &_prime_vector);
		_setKernelArg(_init_factors_neg, 2, sizeof(cl_mem), &_ext2_vector);
		_setKernelArg(_init_factors_neg, 3, sizeof(cl_mem), &_kro_vector);

		_generate_factors_neg = _createKernel("generate_factors_neg");
		_setKernelArg(_generate_factors_neg, 0, sizeof(cl_mem), &_prime_count);
		_setKernelArg(_generate_factors_neg, 1, sizeof(cl_mem), &_prime_vector);
		_setKernelArg(_generate_factors_neg, 2, sizeof(cl_mem), &_ext2_vector);
		_setKernelArg(_generate_factors_neg, 3, sizeof(cl_mem), &_factor_count);
		_setKernelArg(_generate_factors_neg, 4, sizeof(cl_mem), &_factor_vector);

		_clear_primes = _createKernel("clear_primes");
		_setKernelArg(_clear_primes, 0, sizeof(cl_mem), &_prime_count);
	}

public:
	void releaseKernels()
	{
#if defined (ocl_debug)
		std::cerr << "Release ocl kernels." << std::endl;
#endif
		_releaseKernel(_generate_primes_pos);
		_releaseKernel(_init_factors_pos);
		_releaseKernel(_generate_factors_pos);
		_releaseKernel(_generate_primes_neg);
		_releaseKernel(_init_factors_neg);
		_releaseKernel(_generate_factors_neg);
		_releaseKernel(_clear_primes);
	}

public:
	void writeKro(const cl_char * const ptr) { _writeBuffer(_kro_vector, ptr, sizeof(cl_char) * 128 * 256); }
	void clearPrimeCount() { const cl_uint zero = 0; _writeBuffer(_prime_count, &zero, sizeof(cl_uint)); }
	cl_uint readPrimeCount() { cl_uint count; _readBuffer(_prime_count, &count, sizeof(cl_uint)); return count; }
	void clearFactorCount() { const cl_uint zero = 0; _writeBuffer(_factor_count, &zero, sizeof(cl_uint)); }
	cl_uint readFactorCount() { cl_uint count; _readBuffer(_factor_count, &count, sizeof(cl_uint)); return count; }
	void readFactors(cl_ulong2 * const ptr, const size_t count) { if (count > 0) _readBuffer(_factor_vector, ptr, sizeof(cl_ulong2) * count); }

public:
	void generatePrimesPos(const size_t count, const uint64_t index)
	{
		const cl_ulong i = cl_ulong(index);
		_setKernelArg(_generate_primes_pos, 3, sizeof(cl_ulong), &i);
		_executeKernel(_generate_primes_pos, count);
	}

	void initFactorsPos(const size_t count) { _executeKernel(_init_factors_pos, count); }
	void generateFactorsPos(const size_t count) { _executeKernel(_generate_factors_pos, count); }

	void generatePrimesNeg(const size_t count, const uint64_t index)
	{
		const cl_ulong i = cl_ulong(index);
		_setKernelArg(_generate_primes_neg, 3, sizeof(cl_ulong), &i);
		_executeKernel(_generate_primes_neg, count);
	}

	void initFactorsNeg(const size_t count) { _executeKernel(_init_factors_neg, count); }
	void generateFactorsNeg(const size_t count) { _executeKernel(_generate_factors_neg, count); }

	void clearPrimes() { _executeKernel(_clear_primes, 1); }
};
