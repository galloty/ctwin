/*
Copyright 2023, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"

typedef cl_uint		uint32;
typedef cl_int		int32;
typedef cl_ulong	uint64;
typedef cl_long		int64;
typedef cl_uint2	uint32_2;
typedef cl_ulong4	uint64_4;

#define VSIZE	64

inline int ilog2_32(const uint32_t n) { return 31 - __builtin_clz(n); }

class engine : public ocl::device
{
private:
	size_t _nsize = 0, _vnsize = 0, _vn_csize = 0;
	cl_mem _x12 = nullptr, _x3 = nullptr;
	cl_mem _wr12 = nullptr, _wr3 = nullptr, _wri12 = nullptr, _wri3 = nullptr, _bb_inv = nullptr, _bs = nullptr, _f = nullptr;
	cl_mem _data = nullptr, _res = nullptr;
	cl_kernel _set = nullptr, _copy = nullptr;
	cl_kernel _square2 = nullptr, _square4 = nullptr, _square8 = nullptr, _square16 = nullptr, _square32 = nullptr;
	cl_kernel _mul2 = nullptr, _mul4 = nullptr;
	cl_kernel _forward2 = nullptr, _forward4 = nullptr, _forward8 = nullptr, _forward16 = nullptr, _forward16_0 = nullptr, _forward32 = nullptr;
	cl_kernel _backward2 = nullptr, _backward4 = nullptr, _backward8 = nullptr, _backward16 = nullptr, _backward16_0 = nullptr, _backward32 = nullptr;;
	cl_kernel _normalize1 = nullptr, _normalize2 = nullptr;
	// cl_kernel _add_throughput = nullptr, _add_latency = nullptr, _sub_throughput = nullptr, _sub_latency = nullptr;
	// cl_kernel _mul_throughput = nullptr, _mul_latency = nullptr, _but_throughput = nullptr, _but_latency = nullptr;

public:
	engine(const ocl::platform & platform, const size_t d, const bool verbose) : ocl::device(platform, d, verbose) {}
	virtual ~engine() {}

public:
	void allocMemory(const size_t size, const size_t csize)
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Alloc gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		const size_t vnsize = VSIZE * size;
		_nsize = size;
		_vnsize = vnsize;
		_vn_csize = vnsize / csize;
		_x12 = _createBuffer(CL_MEM_READ_WRITE, 3 * sizeof(uint32_2) * vnsize);
		_x3 = _createBuffer(CL_MEM_READ_WRITE, 3 * sizeof(uint32) * vnsize);
		_wr12 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * size);
		_wr3 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32) * size);
		_wri12 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * size);
		_wri3 = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32) * size);
		_bb_inv = _createBuffer(CL_MEM_READ_ONLY, sizeof(uint32_2) * VSIZE);
		_bs = _createBuffer(CL_MEM_READ_ONLY, sizeof(int32) * VSIZE);
		_f = _createBuffer(CL_MEM_READ_WRITE, sizeof(int64) * _vn_csize);
		_data = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * 64);
		_res = _createBuffer(CL_MEM_READ_WRITE, sizeof(uint32) * 8);
	}

public:
	void releaseMemory()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Free gpu memory." << std::endl;
		pio::display(ss.str());
#endif
		if (_nsize != 0)
		{
			_releaseBuffer(_x12); _releaseBuffer(_x3);
			_releaseBuffer(_wr12); _releaseBuffer(_wr3); _releaseBuffer(_wri12); _releaseBuffer(_wri3);
			_releaseBuffer(_bb_inv); _releaseBuffer(_bs); _releaseBuffer(_f);
			_releaseBuffer(_data); _releaseBuffer(_res);
			_nsize = _vnsize = _vn_csize = 0;
		}
	}

private:
	void createKernel_square_mul(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_wri3);
		_setKernelArg(kernel, 4, sizeof(cl_mem), &_x12);
		_setKernelArg(kernel, 5, sizeof(cl_mem), &_x3);
	}

	void createKernel_forward(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wr12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wr3);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_x12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_x3);
	}

	void createKernel_backward(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_wri12);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_wri3);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_x12);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_x3);
	}

	void createKernel_normalize(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_bb_inv);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_bs);
		_setKernelArg(kernel, 2, sizeof(cl_mem), &_f);
		_setKernelArg(kernel, 3, sizeof(cl_mem), &_x12);
		_setKernelArg(kernel, 4, sizeof(cl_mem), &_x3);
	}

	void createKernel_bench(cl_kernel & kernel, const char * const name)
	{
		kernel = _createKernel(name);
		_setKernelArg(kernel, 0, sizeof(cl_mem), &_data);
		_setKernelArg(kernel, 1, sizeof(cl_mem), &_res);
	}

public:
	void createKernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Create ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		_set = _createKernel("set");
		_copy = _createKernel("copy");

		createKernel_square_mul(_square2, "square2");
		createKernel_square_mul(_square4, "square4");
		createKernel_square_mul(_square8, "square8");
		createKernel_square_mul(_square16, "square16");
		if (getMaxWorkGroupSize() >= VSIZE * 32 / 4) createKernel_square_mul(_square32, "square32");

		createKernel_square_mul(_mul2, "mul2");
		createKernel_square_mul(_mul4, "mul4");

		createKernel_forward(_forward2, "forward2");
		createKernel_forward(_forward4, "forward4");
		createKernel_forward(_forward8, "forward8");
		createKernel_forward(_forward16, "forward16");
		createKernel_forward(_forward16_0, "forward16_0");
		if (getMaxWorkGroupSize() >= VSIZE * 32 / 4) createKernel_forward(_forward32, "forward32");

		createKernel_backward(_backward2, "backward2");
		createKernel_backward(_backward4, "backward4");
		createKernel_backward(_backward8, "backward8");
		createKernel_backward(_backward16, "backward16");
		createKernel_backward(_backward16_0, "backward16_0");
		if (getMaxWorkGroupSize() >= VSIZE * 32 / 4) createKernel_backward(_backward32, "backward32");

		createKernel_normalize(_normalize1, "normalize1");
		createKernel_normalize(_normalize2, "normalize2");

		// createKernel_bench(_add_throughput, "add_throughput");
		// createKernel_bench(_add_latency, "add_latency");
		// createKernel_bench(_sub_throughput, "sub_throughput");
		// createKernel_bench(_sub_latency, "sub_latency");
		// createKernel_bench(_mul_throughput, "mul_throughput");
		// createKernel_bench(_mul_latency, "mul_latency");
		// createKernel_bench(_but_throughput, "but_throughput");
		// createKernel_bench(_but_latency, "but_latency");
	}

public:
	void releaseKernels()
	{
#if defined(ocl_debug)
		std::ostringstream ss; ss << "Release ocl kernels." << std::endl;
		pio::display(ss.str());
#endif
		_releaseKernel(_set); _releaseKernel(_copy);
		_releaseKernel(_square2); _releaseKernel(_square4); _releaseKernel(_square8); _releaseKernel(_square16);  _releaseKernel(_square32);
		_releaseKernel(_mul2); _releaseKernel(_mul4);
		_releaseKernel(_forward2); _releaseKernel(_forward4); _releaseKernel(_forward8); _releaseKernel(_forward16); _releaseKernel(_forward16_0); _releaseKernel(_forward32);
		_releaseKernel(_backward2); _releaseKernel(_backward4); _releaseKernel(_backward8); _releaseKernel(_backward16); _releaseKernel(_backward16_0); _releaseKernel(_backward32);
		_releaseKernel(_normalize1); _releaseKernel(_normalize2);
		// _releaseKernel(_add_throughput); _releaseKernel(_add_latency); _releaseKernel(_sub_throughput); _releaseKernel(_sub_latency);
		// _releaseKernel(_mul_throughput); _releaseKernel(_mul_latency); _releaseKernel(_but_throughput); _releaseKernel(_but_latency);
	}

public:
	void writeMemory_w(const uint32_2 * const wr12, const uint32 * const wr3, const uint32_2 * const wri12, const uint32 * const wri3)
	{
		const size_t nsize = this->_nsize;
		_writeBuffer(_wr12, wr12, sizeof(uint32_2) * nsize);
		_writeBuffer(_wr3, wr3, sizeof(uint32) * nsize);
		_writeBuffer(_wri12, wri12, sizeof(uint32_2) * nsize);
		_writeBuffer(_wri3, wri3, sizeof(uint32) * nsize);
	}

	void writeMemory_b(const uint32_2 * const bb_inv, const int32 * const bs)
	{
		_writeBuffer(_bb_inv, bb_inv, sizeof(uint32_2) * VSIZE);
		_writeBuffer(_bs, bs, sizeof(int32) * VSIZE);
	}

	void writeMemory_x123(const uint32_2 * const x12, const uint32 * const x3, const size_t num_regs)
	{
		const size_t size = num_regs * this->_vnsize;
		_writeBuffer(_x12, x12, sizeof(uint32_2) * size);
		_writeBuffer(_x3, x3, sizeof(uint32) * size);
	}

public:
	void readMemory_x3(uint32 * const x3) { _readBuffer(_x3, x3, sizeof(uint32) * this->_vnsize); }

	void readMemory_x123(uint32_2 * const x12, uint32 * const x3, const size_t num_regs)
	{
		const size_t size = num_regs * this->_vnsize;
		_readBuffer(_x12, x12, sizeof(uint32_2) * size);
		_readBuffer(_x3, x3, sizeof(uint32) * size);
	}

private:
	void readMemory_res(uint32 * const res) { _readBuffer(_res, res, sizeof(uint32) * 8); }

public:
	void set(const uint32 a)
	{
		_setKernelArg(_set, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_set, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_set, 2, sizeof(uint32), &a);
		_executeKernel(_set, this->_vnsize);
	}

public:
	void copy(const uint32 reg_dst, const uint32 reg_src)
	{
		_setKernelArg(_copy, 0, sizeof(cl_mem), &_x12);
		_setKernelArg(_copy, 1, sizeof(cl_mem), &_x3);
		_setKernelArg(_copy, 2, sizeof(uint32), &reg_dst);
		_setKernelArg(_copy, 3, sizeof(uint32), &reg_src);
		_executeKernel(_copy, this->_vnsize);
	}

private:
	void square2() { _executeKernel(_square2, this->_vnsize / 2); }
	void square4() { _executeKernel(_square4, this->_vnsize / 4); }
	void square8() { _executeKernel(_square8, this->_vnsize / 4, VSIZE * 8 / 4); }
	void square16() { _executeKernel(_square16, this->_vnsize / 4, VSIZE * 16 / 4); }
	void square32() { _executeKernel(_square32, this->_vnsize / 4, VSIZE * 32 / 4); }

	void mul2() { _executeKernel(_mul2, this->_vnsize / 2); }
	void mul4() { _executeKernel(_mul4, this->_vnsize / 4); }

	void set_sm_args(cl_kernel kernel, const uint32 s, const int32 lm)
	{
		const uint32 m = uint32(1) << lm;
		_setKernelArg(kernel, 4, sizeof(uint32), &s);
		_setKernelArg(kernel, 5, sizeof(uint32), &m);
		_setKernelArg(kernel, 6, sizeof(int32), &lm);
	}

	void forward2(const uint32 s, const int32 lm)
	{
		set_sm_args(_forward2, s, lm);
		_executeKernel(_forward2, this->_vnsize / 2);
	}

	void forward4(const uint32 s, const int32 lm)
	{
		set_sm_args(_forward4, s, lm);
		const uint32 reg0 = 0;
		_setKernelArg(_forward4, 7, sizeof(uint32), &reg0);
		_executeKernel(_forward4, this->_vnsize / 4);
	}

	void forward8(const uint32 s, const int32 lm)
	{
		set_sm_args(_forward8, s, lm);
		_executeKernel(_forward8, this->_vnsize / 8);
	}

	void forward16(const uint32 s, const int32 lm)
	{
		set_sm_args(_forward16, s, lm);
		const uint32 reg0 = 0;
		_setKernelArg(_forward16, 7, sizeof(uint32), &reg0);
		_executeKernel(_forward16, this->_vnsize / 4, VSIZE * 16 / 4);
	}

	void forward16_0()
	{
		const uint32 reg0 = 0;
		_setKernelArg(_forward16_0, 4, sizeof(uint32), &reg0);
		_executeKernel(_forward16_0, this->_vnsize / 4, VSIZE * 16 / 4);
	}

	void forward32(const uint32 s, const int32 lm)
	{
		set_sm_args(_forward32, s, lm);
		_executeKernel(_forward32, this->_vnsize / 4, VSIZE * 32 / 4);
	}

	void backward2(const uint32 s, const int32 lm)
	{
		set_sm_args(_backward2, s, lm);
		_executeKernel(_backward2, this->_vnsize / 2);
	}

	void backward4(const uint32 s, const int32 lm)
	{
		set_sm_args(_backward4, s, lm);
		_executeKernel(_backward4, this->_vnsize / 4);
	}

	void backward8(const uint32 s, const int32 lm)
	{
		set_sm_args(_backward8, s, lm);
		_executeKernel(_backward8, this->_vnsize / 8);
	}

	void backward16(const uint32 s, const int32 lm)
	{
		set_sm_args(_backward16, s, lm);
		_executeKernel(_backward16, this->_vnsize / 4, VSIZE * 16 / 4);
	}

	void backward16_0()
	{
		_executeKernel(_backward16_0, this->_vnsize / 4, VSIZE * 16 / 4);
	}

	void backward32(const uint32 s, const int32 lm)
	{
		set_sm_args(_backward32, s, lm);
		_executeKernel(_backward32, this->_vnsize / 4, VSIZE * 32 / 4);
	}

	void forward4mul(const uint32 s, const int32 lm)
	{
		set_sm_args(_forward4, s, lm);
		const uint32 reg0 = 0;
		_setKernelArg(_forward4, 7, sizeof(uint32), &reg0);
		_executeKernel(_forward4, this->_vnsize / 4);
		const uint32 reg1 = 1;
		_setKernelArg(_forward4, 7, sizeof(uint32), &reg1);
		_executeKernel(_forward4, this->_vnsize / 4);
	}

	void forward16mul(const uint32 s, const int32 lm)
	{
		set_sm_args(_forward16, s, lm);
		const uint32 reg0 = 0;
		_setKernelArg(_forward16, 7, sizeof(uint32), &reg0);
		_executeKernel(_forward16, this->_vnsize / 4, VSIZE * 16 / 4);
		const uint32 reg1 = 1;
		_setKernelArg(_forward16, 7, sizeof(uint32), &reg1);
		_executeKernel(_forward16, this->_vnsize / 4, VSIZE * 16 / 4);
	}

	void forward16mul_0()
	{
		const uint32 reg0 = 0;
		_setKernelArg(_forward16_0, 4, sizeof(uint32), &reg0);
		_executeKernel(_forward16_0, this->_vnsize / 4, VSIZE * 16 / 4);
		const uint32 reg1 = 1;
		_setKernelArg(_forward16_0, 4, sizeof(uint32), &reg1);
		_executeKernel(_forward16_0, this->_vnsize / 4, VSIZE * 16 / 4);
	}

	void normalize1(const uint64 dup)
	{
		_setKernelArg(_normalize1, 5, sizeof(uint64), &dup);
		_executeKernel(_normalize1, this->_vn_csize);
	}

	void normalize2()
	{
		_executeKernel(_normalize2, this->_vn_csize);
	}

public:
	void squareDup(const int32 ln, const uint64 & dup)
	{
		static bool first = true;
		const bool verbose = first ? true : false;
		if (first) first = false;
		forward16_0(); if (verbose) std::cout << "forward16_0 1 " << ln << ", ";
		uint32 s = 16; int32 lm = ln - 4;

		if (getMaxWorkGroupSize() >= VSIZE * 32 / 4)
		{
			for (; lm > 5; s *= 32, lm -= 5) { forward32(s, lm - 5); if (verbose) std::cout << "forward32 " << s << " " << lm << ", "; }
			if      (lm == 1) { square2(); if (verbose) std::cout << "square2 " << s << " " << lm << ", "; }
			else if (lm == 2) { square4(); if (verbose) std::cout << "square4 " << s << " " << lm << ", "; }
			else if (lm == 3) { square8(); if (verbose) std::cout << "square8 " << s << " " << lm << ", "; }
			else if (lm == 4) { square16(); if (verbose) std::cout << "square16 " << s << " " << lm << ", "; }
			else if (lm == 5) { square32(); if (verbose) std::cout << "square32 " << s << " " << lm << ", "; }
			for (s /= 32, lm += 5; s >= 16; s /= 32, lm += 5) { backward32(s, lm - 5); if (verbose) std::cout << "backward32 " << s << " " << lm << ", "; }
		}
		else
		{
			for (; lm > 4; s *= 16, lm -= 4) { forward16(s, lm - 4); if (verbose) std::cout << "forward16 "; }
			if      (lm == 1) { square2(); if (verbose) std::cout << "square2 " << s << " " << lm << ", "; }
			else if (lm == 2) { square4(); if (verbose) std::cout << "square4 " << s << " " << lm << ", "; }
			else if (lm == 3) { square8(); if (verbose) std::cout << "square8 " << s << " " << lm << ", "; }
			else if (lm == 4) { square16(); if (verbose) std::cout << "square16 " << s << " " << lm << ", "; }
			for (s /= 16, lm += 4; s > 1; s /= 16, lm += 4) { backward16(s, lm - 4); if (verbose) std::cout << "backward16 " << s << " " << lm << ", "; }
		}

		backward16_0(); if (verbose) std::cout << "backward16_0 1 " << ln << ", ";
		normalize1(dup); if (verbose) std::cout << "normalize1 ";
		normalize2(); if (verbose) std::cout << "normalize2" << std::endl;
	}

	void mul(const int32 ln)
	{
		forward16mul_0();
		uint32 s = 16; int32 lm = ln - 4;
		for (; lm > 4; s *= 16, lm -= 4) forward16mul(s, lm - 4);
		if (lm > 2) forward4mul(s, lm - 2);
		if (lm % 2 != 0) mul2(); else mul4();
		if (lm > 2) backward4(s, lm - 2);
		for (s /= 16, lm += 4; s > 1; s /= 16, lm += 4) backward16(s, lm - 4);
		backward16_0();
		normalize1(0);
		normalize2();
	}

/*	void bench_add_throughput(uint32 * const res)
	{
		_executeKernel(_add_throughput, 1);
 		readMemory_res(res);
	}
	void bench_add_latency(uint32 * const res)
	{
		_executeKernel(_add_latency, 1);
 		readMemory_res(res);
	}
	void bench_sub_throughput(uint32 * const res)
	{
		_executeKernel(_sub_throughput, 1);
 		readMemory_res(res);
	}
	void bench_sub_latency(uint32 * const res)
	{
		_executeKernel(_sub_latency, 1);
 		readMemory_res(res);
	}
	void bench_mul_throughput(uint32 * const res)
	{
		_executeKernel(_mul_throughput, 1);
 		readMemory_res(res);
	}
	void bench_mul_latency(uint32 * const res)
	{
		_executeKernel(_mul_latency, 1);
 		readMemory_res(res);
	}
	void bench_but_throughput(uint32 * const res)
	{
		_executeKernel(_but_throughput, 1);
 		readMemory_res(res);
	}
	void bench_but_latency(uint32 * const res)
	{
		_executeKernel(_but_latency, 1);
 		readMemory_res(res);
	} */
};
