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
	cl_mem _kro_vector = nullptr;
	cl_mem _prime_vector = nullptr, _prime_3_4_vector = nullptr, _prime_5_8_vector = nullptr;
	cl_mem _ext_vector = nullptr, _ext_3_4_vector = nullptr, _ext_5_8_vector = nullptr;
	cl_mem _factor_vector = nullptr;
	cl_mem _prime_count = nullptr, _prime_3_4_count = nullptr, _prime_5_8_count = nullptr, _factor_count = nullptr;

	cl_kernel _generate_primes = nullptr, _init_factors = nullptr, _generate_factors = nullptr, _clear_primes = nullptr;
	cl_kernel _init_3_4_factors = nullptr, _init_5_8_factors = nullptr, _generate_3_4_factors = nullptr, _generate_5_8_factors = nullptr;

public:
	engine(const platform & platform, const size_t d) : device(platform, d, true) {}
	virtual ~engine() {}

public:
	void allocMemory(const size_t prime_size, const size_t factor_size, const int mode)
	{
#if defined (ocl_debug)
		std::cerr << "Alloc gpu memory." << std::endl;
#endif
		_kro_vector = _createBuffer(CL_MEM_READ_ONLY, sizeof(cl_char) * 128 * 256);
		_prime_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);	// p, q
		if (mode > 0)
		{
			_ext_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong) * prime_size);		// one / a^k
		}
		if (mode < 0)
		{
			_prime_3_4_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);	// p = 3 (mod 4)
			_prime_5_8_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);	// p = 5 (mod 8)
			_ext_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);			// (one, 0) / (b_1, b_2)
			_ext_3_4_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);
			_ext_5_8_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * prime_size);
		}
		_factor_vector = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_ulong2) * factor_size);	// p, b

		_prime_count = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
		if (mode < 0)
		{
			_prime_3_4_count = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
			_prime_5_8_count = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
		}
		_factor_count = _createBuffer(CL_MEM_READ_WRITE, sizeof(cl_uint));
	}

public:
	void releaseMemory()
	{
#if defined (ocl_debug)
		std::cerr << "Free gpu memory." << std::endl;
#endif
		if (_kro_vector != nullptr) _releaseBuffer(_kro_vector);
		if (_prime_vector != nullptr) _releaseBuffer(_prime_vector);
		if (_prime_3_4_vector != nullptr) _releaseBuffer(_prime_3_4_vector);
		if (_prime_5_8_vector != nullptr) _releaseBuffer(_prime_5_8_vector);
		if (_ext_vector != nullptr) _releaseBuffer(_ext_vector);
		if (_ext_3_4_vector != nullptr) _releaseBuffer(_ext_3_4_vector);
		if (_ext_5_8_vector != nullptr) _releaseBuffer(_ext_5_8_vector);
		if (_factor_vector != nullptr) _releaseBuffer(_factor_vector);
		if (_prime_count != nullptr) _releaseBuffer(_prime_count);
		if (_prime_3_4_count != nullptr) _releaseBuffer(_prime_3_4_count);
		if (_prime_5_8_count != nullptr) _releaseBuffer(_prime_5_8_count);
		if (_factor_count != nullptr) _releaseBuffer(_factor_count);
	}

public:
	void createKernels(const int mode)
	{
#if defined (ocl_debug)
		std::cerr << "Create ocl kernels." << std::endl;
#endif
		if (mode > 0)
		{
			_generate_primes = _createKernel("generate_primes_pos");
			_setKernelArg(_generate_primes, 0, sizeof(cl_mem), &_prime_count);
			_setKernelArg(_generate_primes, 1, sizeof(cl_mem), &_prime_vector);
			_setKernelArg(_generate_primes, 2, sizeof(cl_mem), &_ext_vector);

			_init_factors = _createKernel("init_factors_pos");
			_setKernelArg(_init_factors, 0, sizeof(cl_mem), &_prime_count);
			_setKernelArg(_init_factors, 1, sizeof(cl_mem), &_prime_vector);
			_setKernelArg(_init_factors, 2, sizeof(cl_mem), &_ext_vector);
			_setKernelArg(_init_factors, 3, sizeof(cl_mem), &_kro_vector);

			_generate_factors = _createKernel("generate_factors_pos");
			_setKernelArg(_generate_factors, 0, sizeof(cl_mem), &_prime_count);
			_setKernelArg(_generate_factors, 1, sizeof(cl_mem), &_prime_vector);
			_setKernelArg(_generate_factors, 2, sizeof(cl_mem), &_ext_vector);
			_setKernelArg(_generate_factors, 3, sizeof(cl_mem), &_factor_count);
			_setKernelArg(_generate_factors, 4, sizeof(cl_mem), &_factor_vector);

			_clear_primes = _createKernel("clear_primes_pos");
			_setKernelArg(_clear_primes, 0, sizeof(cl_mem), &_prime_count);
		}
		if (mode < 0)
		{
			_generate_primes = _createKernel("generate_primes_neg");
			_setKernelArg(_generate_primes, 0, sizeof(cl_mem), &_prime_count);
			_setKernelArg(_generate_primes, 1, sizeof(cl_mem), &_prime_vector);
			_setKernelArg(_generate_primes, 2, sizeof(cl_mem), &_ext_vector);
			_setKernelArg(_generate_primes, 3, sizeof(cl_mem), &_prime_3_4_count);
			_setKernelArg(_generate_primes, 4, sizeof(cl_mem), &_prime_3_4_vector);
			_setKernelArg(_generate_primes, 5, sizeof(cl_mem), &_ext_3_4_vector);
			_setKernelArg(_generate_primes, 6, sizeof(cl_mem), &_prime_5_8_count);
			_setKernelArg(_generate_primes, 7, sizeof(cl_mem), &_prime_5_8_vector);
			_setKernelArg(_generate_primes, 8, sizeof(cl_mem), &_ext_5_8_vector);

			_init_factors = _createKernel("init_factors_neg");
			_setKernelArg(_init_factors, 0, sizeof(cl_mem), &_prime_count);
			_setKernelArg(_init_factors, 1, sizeof(cl_mem), &_prime_vector);
			_setKernelArg(_init_factors, 2, sizeof(cl_mem), &_ext_vector);

			_generate_factors = _createKernel("generate_factors_neg");
			_setKernelArg(_generate_factors, 0, sizeof(cl_mem), &_prime_count);
			_setKernelArg(_generate_factors, 1, sizeof(cl_mem), &_prime_vector);
			_setKernelArg(_generate_factors, 2, sizeof(cl_mem), &_ext_vector);
			_setKernelArg(_generate_factors, 3, sizeof(cl_mem), &_factor_count);
			_setKernelArg(_generate_factors, 4, sizeof(cl_mem), &_factor_vector);

			_clear_primes = _createKernel("clear_primes_neg");
			_setKernelArg(_clear_primes, 0, sizeof(cl_mem), &_prime_count);
			_setKernelArg(_clear_primes, 1, sizeof(cl_mem), &_prime_3_4_count);
			_setKernelArg(_clear_primes, 2, sizeof(cl_mem), &_prime_5_8_count);

			_init_3_4_factors = _createKernel("init_3_4_factors_neg");
			_setKernelArg(_init_3_4_factors, 0, sizeof(cl_mem), &_prime_3_4_count);
			_setKernelArg(_init_3_4_factors, 1, sizeof(cl_mem), &_prime_3_4_vector);
			_setKernelArg(_init_3_4_factors, 2, sizeof(cl_mem), &_ext_3_4_vector);

			_init_5_8_factors = _createKernel("init_5_8_factors_neg");
			_setKernelArg(_init_5_8_factors, 0, sizeof(cl_mem), &_prime_5_8_count);
			_setKernelArg(_init_5_8_factors, 1, sizeof(cl_mem), &_prime_5_8_vector);
			_setKernelArg(_init_5_8_factors, 2, sizeof(cl_mem), &_ext_5_8_vector);

			_generate_3_4_factors = _createKernel("generate_factors_neg");
			_setKernelArg(_generate_3_4_factors, 0, sizeof(cl_mem), &_prime_3_4_count);
			_setKernelArg(_generate_3_4_factors, 1, sizeof(cl_mem), &_prime_3_4_vector);
			_setKernelArg(_generate_3_4_factors, 2, sizeof(cl_mem), &_ext_3_4_vector);
			_setKernelArg(_generate_3_4_factors, 3, sizeof(cl_mem), &_factor_count);
			_setKernelArg(_generate_3_4_factors, 4, sizeof(cl_mem), &_factor_vector);

			_generate_5_8_factors = _createKernel("generate_factors_neg");
			_setKernelArg(_generate_5_8_factors, 0, sizeof(cl_mem), &_prime_5_8_count);
			_setKernelArg(_generate_5_8_factors, 1, sizeof(cl_mem), &_prime_5_8_vector);
			_setKernelArg(_generate_5_8_factors, 2, sizeof(cl_mem), &_ext_5_8_vector);
			_setKernelArg(_generate_5_8_factors, 3, sizeof(cl_mem), &_factor_count);
			_setKernelArg(_generate_5_8_factors, 4, sizeof(cl_mem), &_factor_vector);
		}
	}

public:
	void releaseKernels()
	{
#if defined (ocl_debug)
		std::cerr << "Release ocl kernels." << std::endl;
#endif
		if (_generate_primes != nullptr) _releaseKernel(_generate_primes);
		if (_init_factors != nullptr) _releaseKernel(_init_factors);
		if (_generate_factors != nullptr) _releaseKernel(_generate_factors);
		if (_clear_primes != nullptr) _releaseKernel(_clear_primes);
		if (_init_3_4_factors != nullptr) _releaseKernel(_init_3_4_factors);
		if (_init_5_8_factors != nullptr) _releaseKernel(_init_5_8_factors);
		if (_generate_3_4_factors != nullptr) _releaseKernel(_generate_3_4_factors);
		if (_generate_5_8_factors != nullptr) _releaseKernel(_generate_5_8_factors);
	}

public:
	void writeKro(const cl_char * const ptr) { _writeBuffer(_kro_vector, ptr, sizeof(cl_char) * 128 * 256); }
	// cl_uint readPrimeCount() { cl_uint count; _readBuffer(_prime_count, &count, sizeof(cl_uint)); return count; }
	void clearFactorCount() { const cl_uint zero = 0; _writeBuffer(_factor_count, &zero, sizeof(cl_uint)); }
	cl_uint readFactorCount() { cl_uint count; _readBuffer(_factor_count, &count, sizeof(cl_uint)); return count; }
	void readFactors(cl_ulong2 * const ptr, const size_t count) { if (count > 0) _readBuffer(_factor_vector, ptr, sizeof(cl_ulong2) * count); }

public:
	void generatePrimesPos(const size_t count, const uint64_t index)
	{
		const cl_ulong i = cl_ulong(index);
		_setKernelArg(_generate_primes, 3, sizeof(cl_ulong), &i);
		_executeKernel(_generate_primes, count);
	}

	void initFactorsPos(const size_t count) { _executeKernel(_init_factors, count); }
	void generateFactorsPos(const size_t count) { _executeKernel(_generate_factors, count); }
	void clearPrimesPos() { _executeKernel(_clear_primes, 1); }

	void generatePrimesNeg(const size_t count, const uint64_t index)
	{
		const cl_ulong i = cl_ulong(index);
		_setKernelArg(_generate_primes, 9, sizeof(cl_ulong), &i);
		_executeKernel(_generate_primes, count);
	}

	void initFactorsNeg(const size_t count)
	{
		_executeKernel(_init_3_4_factors, count);
		_executeKernel(_init_5_8_factors, count);
		_executeKernel(_init_factors, count);
	}

	void generateFactorsNeg(const size_t count)
	{
		_executeKernel(_generate_3_4_factors, count);
		_executeKernel(_generate_5_8_factors, count);
		_executeKernel(_generate_factors, count);
	}

	void clearPrimesNeg() { _executeKernel(_clear_primes, 1); }
};
