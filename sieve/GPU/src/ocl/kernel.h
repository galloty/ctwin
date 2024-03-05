/*
Copyright 2024, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel = \
"/*\n" \
"Copyright 2024, Yves Gallot\n" \
"\n" \
"ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"typedef uint	uint32;\n" \
"typedef ulong	uint64;\n" \
"\n" \
"inline int jacobi(const uint32 x, const uint32 y)\n" \
"{\n" \
"	uint32 m = x, n = y;\n" \
"\n" \
"	int k = 1;\n" \
"	while (m != 0)\n" \
"	{\n" \
"		// (2/n) = (-1)^((n^2-1)/8)\n" \
"		bool odd = false;\n" \
"		while (m % 2 == 0) { m /= 2; odd = !odd; }\n" \
"		if (odd && (n % 8 != 1) && (n % 8 != 7)) k = -k;\n" \
"\n" \
"		if (m == 1) return k;	// (1/n) = 1\n" \
"\n" \
"		// (m/n)(n/m) = -1 iif m == n == 3 (mod 4)\n" \
"		if ((m % 4 == 3) && (n % 4 == 3)) k = -k;\n" \
"		const uint32 t = n; n = m; m = t;\n" \
"\n" \
"		m %= n;	// (m/n) = (m mod n / n)\n" \
"	}\n" \
"\n" \
"	return n;	// x and y are not coprime, return their gcd\n" \
"}\n" \
"\n" \
"inline int uint64_log2(const uint64 x) { return 63 - clz(x); }\n" \
"\n" \
"inline uint64 REDC(const uint64 t_lo, const uint64 t_hi, const uint64 p, const uint64 q)\n" \
"{\n" \
"	const uint64 mp = mul_hi(t_lo * q, p), r = t_hi - mp;\n" \
"	return (t_hi < mp) ? r + p : r;\n" \
"}\n" \
"\n" \
"inline uint64 REDCshort(const uint64 t, const uint64 p, const uint64 q)\n" \
"{\n" \
"	const uint64 mp = mul_hi(t * q, p);\n" \
"	return (mp != 0) ? p - mp : 0;\n" \
"}\n" \
"\n" \
"inline uint64 add_mod(const uint64 a, const uint64 b, const uint64 p)\n" \
"{\n" \
"	const uint64 c = (a >= p - b) ? p : 0;\n" \
"	return a + b - c;\n" \
"}\n" \
"\n" \
"inline uint64 mul_mod(const uint64 a, const uint64 b, const uint64 p, const uint64 q)\n" \
"{\n" \
"	return REDC(a * b, mul_hi(a, b), p, q);\n" \
"}\n" \
"\n" \
"// Montgomery form of 2^64 is (2^64)^2\n" \
"inline uint64 two_pow_64(const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	uint64 t = add_mod(one, one, p); t = add_mod(t, t, p);		// 4\n" \
"	t = add_mod(t, t, p); t = add_mod(t, t, p);					// 16\n" \
"	for (size_t i = 0; i < 4; ++i) t = mul_mod(t, t, p, q);		// 16^{2^4} = 2^64\n" \
"	return t;\n" \
"}\n" \
"\n" \
"inline uint64 toMp(const uint64 n, const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	const uint64 r2 = two_pow_64(p, q, one);\n" \
"	return mul_mod(n, r2, p, q);\n" \
"}\n" \
"\n" \
"inline uint64 toInt(const uint64 r, const uint64 p, const uint64 q)\n" \
"{\n" \
"	return REDCshort(r, p, q);\n" \
"}\n" \
"\n" \
"// p * p_inv = 1 (mod 2^64) (Newton's method)\n" \
"inline uint64 invert(const uint64 p)\n" \
"{\n" \
"	uint64 p_inv = 1, prev = 0;\n" \
"	while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }\n" \
"	return p_inv;\n" \
"}\n" \
"\n" \
"// a^e mod p, left-to-right algorithm\n" \
"inline uint64 pow_mod(const uint64 a, const uint64 e, const uint64 p, const uint64 q)\n" \
"{\n" \
"	uint64 r = a;\n" \
"	for (int b = uint64_log2(e) - 1; b >= 0; --b)\n" \
"	{\n" \
"		r = mul_mod(r, r, p, q);\n" \
"		if ((e & ((uint64)(1) << b)) != 0) r = mul_mod(r, a, p, q);\n" \
"	}\n" \
"	return r;\n" \
"}\n" \
"\n" \
"// 2^(p - 1) ?= 1 mod p\n" \
"inline bool prp(const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	const uint64 e = (p - 1) / 2;\n" \
"	int b = uint64_log2(e) - 1;\n" \
"	uint64 r = add_mod(one, one, p); r = add_mod(r, r, p);\n" \
"	if ((e & ((uint64)(1) << b)) != 0) r = add_mod(r, r, p);\n" \
"	for (--b; b >= 0; --b)\n" \
"	{\n" \
"		r = mul_mod(r, r, p, q);\n" \
"		if ((e & ((uint64)(1) << b)) != 0) r = add_mod(r, r, p);\n" \
"	}\n" \
"	return ((r == one) || (r == p - one));\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void check_primes(__global uint * restrict const prime_count, __global ulong3 * restrict const prime_vector, const ulong i)\n" \
"{\n" \
"	const uint64 k = (i << log2GlobalWorkSize) | get_global_id(0);\n" \
"\n" \
"	const uint64 p = (k << (gfn_n + 1)) | 1, q = invert(p), one = (-p) % p;\n" \
"	if (prp(p, q, one))\n" \
"	{\n" \
"		const uint prime_index = atomic_inc(prime_count);\n" \
"		prime_vector[prime_index] = (ulong3)(p, q, one);\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void init_factors(__global const uint * restrict const prime_count, __global const ulong3 * restrict const prime_vector,\n" \
"	__global ulong2 * restrict const c_a2k_vector)\n" \
"{\n" \
"	const size_t i = get_global_id(0);\n" \
"	if (i >= *prime_count) return;\n" \
"\n" \
"	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1, one = prime_vector[i].s2;\n" \
"\n" \
"	const uint64 k = p >> (gfn_n + 1);\n" \
"\n" \
"	uint32 a;\n" \
"	if ((p % 3) == 2) { a = 3; }\n" \
"	else\n" \
"	{\n" \
"		const uint32 pmod5 = p % 5;\n" \
"		if ((pmod5 == 2) || (pmod5 == 3)) { a = 5; }\n" \
"		else\n" \
"		{\n" \
"			const uint32 pmod7 = p % 7;\n" \
"			if ((pmod7 == 3) || (pmod7 == 5) || (pmod7 == 6)) { a = 7; }\n" \
"			else\n" \
"			{\n" \
"				for (a = 11; a < 256; a += 2)\n" \
"				{\n" \
"					const uint32 pmoda = p % a;\n" \
"					if (jacobi(pmoda, a) == -1) break;\n" \
"				}\n" \
"				if (a >= 256) return;	// error?\n" \
"			}\n" \
"		}\n" \
"	}\n" \
"	uint64 ma = toMp(a, p, q, one);\n" \
"\n" \
"	const uint64 c = pow_mod(ma, k, p, q);\n" \
"	const uint64 a2k = mul_mod(c, c, p, q);\n" \
"	c_a2k_vector[i] = (ulong2)(toInt(c, p, q), a2k);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void check_factors(__global const uint * restrict const prime_count, __global const ulong3 * restrict const prime_vector,\n" \
"	__global const ulong2 * restrict const c_a2k_vector, __global uint * restrict const factor_count, __global ulong2 * restrict const factor)\n" \
"{\n" \
"	const size_t i = get_global_id(0);\n" \
"	if (i >= *prime_count) return;\n" \
"\n" \
"	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;\n" \
"	uint64 c = c_a2k_vector[i].s0;\n" \
"	const uint64 a2k = c_a2k_vector[i].s1;\n" \
"\n" \
"	for (size_t i = 0; i < factors_loop; ++i)\n" \
"	{\n" \
"		if (c % 2 != 0) c = p - c;\n" \
"		if (c <= 2000000000)\n" \
"		{\n" \
"			const uint factor_index = atomic_inc(factor_count);\n" \
"			factor[factor_index] = (ulong2)(p, c);\n" \
"		}\n" \
"		c = mul_mod(c, a2k, p, q);		// c = a^{(2*i + 1).k}\n" \
"	}\n" \
"\n" \
"	c_a2k_vector[i].s0 = c;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void clear_primes(__global uint * restrict const prime_count)\n" \
"{\n" \
"	*prime_count = 0;\n" \
"}\n" \
"";
