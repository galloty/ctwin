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
"/*inline uint64 two_pow_64(const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	uint64 t = add_mod(one, one, p); t = add_mod(t, t, p);		// 4\n" \
"	t = add_mod(t, t, p); t = add_mod(t, t, p);					// 16\n" \
"	for (size_t i = 0; i < 4; ++i) t = mul_mod(t, t, p, q);		// 16^{2^4} = 2^64\n" \
"	return t;\n" \
"}*/\n" \
"\n" \
"/*inline uint64 toMp(const uint64 n, const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	const uint64 r2 = two_pow_64(p, q, one);\n" \
"	return mul_mod(n, r2, p, q);\n" \
"}*/\n" \
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
"void generate_primes(__global uint * restrict const prime_count, __global ulong2 * restrict const prime_vector,\n" \
"	__global ulong * restrict const ext_vector, const ulong index)\n" \
"{\n" \
"	const uint64 k = index | get_global_id(0);\n" \
"\n" \
"	const uint64 p = 3 * (k << g_n) + 1, q = invert(p), one = (-p) % p;\n" \
"	if (prp(p, q, one))\n" \
"	{\n" \
"		const uint prime_index = atomic_inc(prime_count);\n" \
"		prime_vector[prime_index] = (ulong2)(p, q);\n" \
"		ext_vector[prime_index] = one;\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void init_factors(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,\n" \
"	__global ulong * restrict const ext_vector, __global const char * restrict const kro_vector)\n" \
"{\n" \
"	const size_t i = get_global_id(0);\n" \
"	if (i >= *prime_count) return;\n" \
"\n" \
"	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;\n" \
"	const uint64 one = ext_vector[i], two = add_mod(one, one, p), four = add_mod(two, two, p);\n" \
"\n" \
"	const uint64 pm1_6 = (p - 1) / 6;\n" \
"\n" \
"	// p = 1 (mod 4). If a is odd then (a/p) = (p/a) = ({p mod a}/a)\n" \
"\n" \
"	uint32 a = 5; uint64 ma = add_mod(four, one, p);\n" \
"	while (a < 256)\n" \
"	{\n" \
"		if ((kro_vector[(a - 5) * 128 + (p % a)] != 0) && (pow_mod(ma, pm1_6, p, q) != p - one)) break;\n" \
"\n" \
"		a += 2; ma = add_mod(ma, two, p);\n" \
"\n" \
"		if ((kro_vector[(a - 5) * 128 + (p % a)] != 0) && (pow_mod(ma, pm1_6, p, q) != p - one)) break;\n" \
"\n" \
"		a += 4; ma = add_mod(ma, four, p);\n" \
"	}\n" \
"\n" \
"	// We have a^{3.k.2^{n-1}} = -1 and a^{k.2^{n-1}} != -1.\n" \
"	// Then x = (a^k)^{2^{n-1}} = j or 1/j and x^2 - x + 1 = 0.\n" \
"\n" \
"	ext_vector[i] = (a < 256) ? pow_mod(ma, pm1_6 >> (g_n - 1), p, q) : (ulong)(0);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void generate_factors(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,\n" \
"	__global const ulong * restrict const ext_vector, __global uint * restrict const factor_count, __global ulong2 * restrict const factor)\n" \
"{\n" \
"	const size_t i = get_global_id(0);\n" \
"	if (i >= *prime_count) return;\n" \
"\n" \
"	uint64 b = ext_vector[i];\n" \
"	if (b == 0) return;\n" \
"\n" \
"	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;\n" \
"	const uint64 b2 = mul_mod(b, b, p, q), b4 = mul_mod(b2, b2, p, q);\n" \
"	b = toInt(b, p, q);\n" \
"\n" \
"	for (uint32 j = 1; j < (3u << (g_n - 1)); j += 6)\n" \
"	{\n" \
"		if (b <= 10 * 1000000000ul)\n" \
"		{\n" \
"			const uint factor_index = atomic_inc(factor_count);\n" \
"			factor[factor_index] = (ulong2)(p, b);\n" \
"		}\n" \
"\n" \
"		if (p - b <= 10 * 1000000000ul)\n" \
"		{\n" \
"			const uint factor_index = atomic_inc(factor_count);\n" \
"			factor[factor_index] = (ulong2)(p, p - b);\n" \
"		}\n" \
"\n" \
"		b = mul_mod(b, b4, p, q);		// b = (a^k)^{j + 4}\n" \
"\n" \
"		if (b <= 10 * 1000000000ul)\n" \
"		{\n" \
"			const uint factor_index = atomic_inc(factor_count);\n" \
"			factor[factor_index] = (ulong2)(p, b);\n" \
"		}\n" \
"\n" \
"		if (p - b <= 10 * 1000000000ul)\n" \
"		{\n" \
"			const uint factor_index = atomic_inc(factor_count);\n" \
"			factor[factor_index] = (ulong2)(p, p - b);\n" \
"		}\n" \
"\n" \
"		b = mul_mod(b, b2, p, q);		// b = (a^k)^{j + 6}\n" \
"	}\n" \
"\n" \
"	// ext_vector[i] = b;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void clear_primes(__global uint * restrict const prime_count)\n" \
"{\n" \
"	*prime_count = 0;\n" \
"}\n" \
"";
