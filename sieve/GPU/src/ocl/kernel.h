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
"inline uint64 sub_mod(const uint64 a, const uint64 b, const uint64 p)\n" \
"{\n" \
"	const uint64 c = (a < b) ? p : 0;\n" \
"	return a - b + c;\n" \
"}\n" \
"\n" \
"inline uint64 half_mod(const uint64 a, const uint64 p)\n" \
"{\n" \
"	return (a % 2 == 0) ? a / 2 : a / 2 + p / 2 + 1;\n" \
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
"}\n" \
"\n" \
"inline uint64 toMp(const uint64 n, const uint64 p, const uint64 q, const uint64 one)\n" \
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
"inline int jacobi(const uint64 x, const uint64 y)\n" \
"{\n" \
"	uint64 m = x, n = y;\n" \
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
"		const uint64 t = n; n = m; m = t;\n" \
"\n" \
"		m %= n;	// (m/n) = (m mod n / n)\n" \
"	}\n" \
"\n" \
"	return 0;	// x and y are not coprime\n" \
"}\n" \
"\n" \
"// sqrt for p = 3 (mod 4)\n" \
"inline uint64 sqrt_mod_3_4(const uint64 a, const uint64 p, const uint64 q)\n" \
"{\n" \
"	return pow_mod(a, (p + 1) / 4, p, q);\n" \
"}\n" \
"\n" \
"// sqrt for p = 5 (mod 8)\n" \
"inline uint64 sqrt_mod_5_8(const uint64 a, const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	const uint64 b = add_mod(a, a, p);\n" \
"	const uint64 v = pow_mod(b, (p - 5) / 8, p, q);\n" \
"	const uint64 i = mul_mod(b, mul_mod(v, v, p, q), p, q);	// i^2 = -1\n" \
"	return mul_mod(mul_mod(a, v, p, q), sub_mod(i, one, p), p, q);\n" \
"}\n" \
"\n" \
"// Cipolla's algorithm\n" \
"inline uint64 sqrt_mod(const uint64 a, const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	if ((a == 0) || (a == one)) return a;\n" \
"\n" \
"	uint32 d = 2; uint64 md = add_mod(one, one, p), ai = p - toInt(a, p, q);\n" \
"	while (d < 256)\n" \
"	{\n" \
"		if (jacobi(d * d + ai, p) == -1) break;\n" \
"		++d; md = add_mod(md, one, p);\n" \
"	}\n" \
"	if (d >= 256) return 0;\n" \
"\n" \
"	const uint64 w2 = sub_mod(mul_mod(md, md, p, q), a, p), e = (p + 1) / 2;\n" \
"	uint64 x = md, y = one;\n" \
"	for (int b = uint64_log2(e) - 1; b >= 0; --b)\n" \
"	{\n" \
"		const uint64 t = add_mod(mul_mod(x, x, p, q), mul_mod(mul_mod(y, y, p, q), w2, p, q), p); y = mul_mod(add_mod(x, x, p), y, p, q); x = t;\n" \
"		if ((e & ((uint64)(1) << b)) != 0)\n" \
"		{\n" \
"			const uint64 t = add_mod(mul_mod(x, md, p, q), mul_mod(y, w2, p, q), p); y = add_mod(x, mul_mod(y, md, p, q), p); x = t;\n" \
"		}\n" \
"	}\n" \
"\n" \
"	return ((mul_mod(x, x, p, q) == a) ? x : 0);\n" \
"}\n" \
"\n" \
"inline uint64 sqrtn_3_4(const uint64 a, const uint32 n, const uint64 p, const uint64 q)\n" \
"{\n" \
"	uint64 r = a, k = (p - 1) / 2, m = 1;\n" \
"	for (uint32 i = 1; i <= n; ++i)\n" \
"	{\n" \
"		if (m % 2 != 0) m += k;\n" \
"		m /= 2;\n" \
"	}\n" \
"	if (m > 1) r = pow_mod(r, m, p, q);\n" \
"\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint64 sqrtn_5_8(const uint64 a, const uint32 n, const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	uint32 e = 1; for (uint64 k = (p - 1) / 2; k % 2 == 0; k /= 2) ++e;\n" \
"	e = min(e, n);\n" \
"	if ((e > 1) && pow_mod(a, (p - 1) >> e, p, q) != one) return 0;\n" \
"\n" \
"	uint64 r = a, k = (p - 1) / 2, m = 1;\n" \
"	for (uint32 i = 1; i <= n; ++i)\n" \
"	{\n" \
"		if (k % 2 == 0)\n" \
"		{\n" \
"			r = sqrt_mod_5_8(r, p, q, one);\n" \
"			k /= 2;\n" \
"			if (r == 0) break;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			if (m % 2 != 0) m += k;\n" \
"			m /= 2;\n" \
"		}\n" \
"	}\n" \
"	if (m > 1) r = pow_mod(r, m, p, q);\n" \
"\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint64 sqrtn(const uint64 a, const uint32 n, const uint64 p, const uint64 q, const uint64 one)\n" \
"{\n" \
"	uint32 e = 1; for (uint64 k = (p - 1) / 2; k % 2 == 0; k /= 2) ++e;\n" \
"	e = min(e, n);\n" \
"	if ((e > 1) && pow_mod(a, (p - 1) >> e, p, q) != one) return 0;\n" \
"\n" \
"	uint64 r = a, k = (p - 1) / 2, m = 1;\n" \
"	for (uint32 i = 1; i <= n; ++i)\n" \
"	{\n" \
"		if (k % 2 == 0)\n" \
"		{\n" \
"			r = sqrt_mod(r, p, q, one);\n" \
"			k /= 2;\n" \
"			if (r == 0) break;\n" \
"		}\n" \
"		else\n" \
"		{\n" \
"			if (m % 2 != 0) m += k;\n" \
"			m /= 2;\n" \
"		}\n" \
"	}\n" \
"	if (m > 1) r = pow_mod(r, m, p, q);\n" \
"\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline void push_factor(const uint64 b, uint64 bMax, const uint64 p,\n" \
"	__global uint * restrict const factor_count, __global ulong2 * restrict const factor)\n" \
"{\n" \
"	if (b <= bMax)\n" \
"	{\n" \
"		const uint factor_index = atomic_inc(factor_count);\n" \
"		factor[factor_index] = (ulong2)(p, b);\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void generate_primes_pos(__global uint * restrict const prime_count, __global ulong2 * restrict const prime_vector,\n" \
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
"void init_factors_pos(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,\n" \
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
"void generate_factors_pos(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,\n" \
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
"	const uint64 bMax = (p < 10000000000000000ul) ? 5 * 1000000000ul : 10 * 1000000000ul;\n" \
"\n" \
"	for (uint32 j = 1; j < (3u << (g_n - 1)); j += 6)\n" \
"	{\n" \
"		push_factor(b, bMax, p, factor_count, factor);\n" \
"		push_factor(p - b, bMax, p, factor_count, factor);\n" \
"\n" \
"		b = mul_mod(b, b4, p, q);		// b = (a^k)^{j + 4}\n" \
"\n" \
"		push_factor(b, bMax, p, factor_count, factor);\n" \
"		push_factor(p - b, bMax, p, factor_count, factor);\n" \
"\n" \
"		b = mul_mod(b, b2, p, q);		// b = (a^k)^{j + 6}\n" \
"	}\n" \
"\n" \
"	// ext_vector[i] = b;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void clear_primes_pos(__global uint * restrict const prime_count)\n" \
"{\n" \
"	*prime_count = 0;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void generate_primes_neg(__global uint * restrict const prime_count, __global ulong2 * restrict const prime_vector, __global ulong2 * restrict const ext_vector,\n" \
"	__global uint * restrict const prime_3_4_count, __global ulong2 * restrict const prime_3_4_vector, __global ulong2 * restrict const ext_3_4_vector,\n" \
"	__global uint * restrict const prime_5_8_count, __global ulong2 * restrict const prime_5_8_vector, __global ulong2 * restrict const ext_5_8_vector,\n" \
"	const ulong index)\n" \
"{\n" \
"	const uint64 k = index | get_global_id(0);\n" \
"\n" \
"	for (uint32 i = 0; i <= 1; ++i)\n" \
"	{\n" \
"		const uint64 p = 10 * k + 2 * i - 1, q = invert(p), one = (-p) % p;\n" \
"		if (prp(p, q, one))\n" \
"		{\n" \
"			if (p % 4 == 3)\n" \
"			{\n" \
"				const uint prime_index = atomic_inc(prime_3_4_count);\n" \
"				prime_3_4_vector[prime_index] = (ulong2)(p, q);\n" \
"				ext_3_4_vector[prime_index] = (ulong2)(one, 0);\n" \
"			}\n" \
"			else if (p % 8 == 5)\n" \
"			{\n" \
"				const uint prime_index = atomic_inc(prime_5_8_count);\n" \
"				prime_5_8_vector[prime_index] = (ulong2)(p, q);\n" \
"				ext_5_8_vector[prime_index] = (ulong2)(one, 0);\n" \
"			}\n" \
"			else\n" \
"			{\n" \
"				const uint prime_index = atomic_inc(prime_count);\n" \
"				prime_vector[prime_index] = (ulong2)(p, q);\n" \
"				ext_vector[prime_index] = (ulong2)(one, 0);\n" \
"			}\n" \
"		}\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void init_3_4_factors_neg(__global const uint * restrict const prime_3_4_count, __global const ulong2 * restrict const prime_3_4_vector,\n" \
"	__global ulong2 * restrict const ext_3_4_vector)\n" \
"{\n" \
"	const size_t i = get_global_id(0);\n" \
"	if (i >= *prime_3_4_count) return;\n" \
"\n" \
"	const uint64 p = prime_3_4_vector[i].s0, q = prime_3_4_vector[i].s1;\n" \
"	const uint64 one = ext_3_4_vector[i].s0, two = add_mod(one, one, p), five = add_mod(add_mod(two, two, p), one, p);\n" \
"\n" \
"	const uint64 s5 = sqrt_mod_3_4(five, p, q);\n" \
"	if (s5 == 0) { ext_3_4_vector[i] = (ulong2)(0, 0); return; }\n" \
"\n" \
"	const uint64 r = half_mod(sub_mod(one, s5, p), p);\n" \
"	const bool is_square = (jacobi(toInt(r, p, q), p) == 1);\n" \
"	const uint64 rs = is_square ? r : sub_mod(one, r, p);\n" \
"	const uint64 sn = sqrtn_3_4(rs, g_n - 1, p, q);\n" \
"	ext_3_4_vector[i] = (ulong2)(sn, 0);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void init_5_8_factors_neg(__global const uint * restrict const prime_5_8_count, __global const ulong2 * restrict const prime_5_8_vector,\n" \
"	__global ulong2 * restrict const ext_5_8_vector)\n" \
"{\n" \
"	const size_t i = get_global_id(0);\n" \
"	if (i >= *prime_5_8_count) return;\n" \
"\n" \
"	const uint64 p = prime_5_8_vector[i].s0, q = prime_5_8_vector[i].s1;\n" \
"	const uint64 one = ext_5_8_vector[i].s0, two = add_mod(one, one, p), five = add_mod(add_mod(two, two, p), one, p);\n" \
"\n" \
"	const uint64 s5 = sqrt_mod_5_8(five, p, q, one);\n" \
"	if (s5 == 0) { ext_5_8_vector[i] = (ulong2)(0, 0); return; }\n" \
"\n" \
"	const uint64 r = half_mod(sub_mod(one, s5, p), p);\n" \
"	if (jacobi(toInt(r, p, q), p) != 1) { ext_5_8_vector[i] = (ulong2)(0, 0); return; }\n" \
"	const uint64 sn1 = sqrtn_5_8(r, g_n - 1, p, q, one);\n" \
"	const uint64 sn2 = sqrtn_5_8(sub_mod(one, r, p), g_n - 1, p, q, one);\n" \
"	ext_5_8_vector[i] = (ulong2)(sn1, sn2);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void init_factors_neg(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,\n" \
"	__global ulong2 * restrict const ext_vector)\n" \
"{\n" \
"	const size_t i = get_global_id(0);\n" \
"	if (i >= *prime_count) return;\n" \
"\n" \
"	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;\n" \
"	const uint64 one = ext_vector[i].s0, two = add_mod(one, one, p), five = add_mod(add_mod(two, two, p), one, p);\n" \
"\n" \
"	const uint64 s5 = sqrt_mod(five, p, q, one);\n" \
"	if (s5 == 0) { ext_vector[i] = (ulong2)(0, 0); return; }\n" \
"\n" \
"	const uint64 r = half_mod(sub_mod(one, s5, p), p);\n" \
"	if (jacobi(toInt(r, p, q), p) != 1) { ext_vector[i] = (ulong2)(0, 0); return; }\n" \
"	const uint64 sn1 = sqrtn(r, g_n - 1, p, q, one);\n" \
"	const uint64 sn2 = sqrtn(sub_mod(one, r, p), g_n - 1, p, q, one);\n" \
"	ext_vector[i] = (ulong2)(sn1, sn2);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void generate_factors_neg(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,\n" \
"	__global const ulong2 * restrict const ext_vector, __global uint * restrict const factor_count, __global ulong2 * restrict const factor)\n" \
"{\n" \
"	const size_t i = get_global_id(0);\n" \
"	if (i >= *prime_count) return;\n" \
"\n" \
"	uint64 r1 = ext_vector[i].s0, r2 = ext_vector[i].s1;\n" \
"\n" \
"	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1, one = (-p) % p;\n" \
"\n" \
"	const uint64 bMax = (p < 10000000000000000ul) ? 5 * 1000000000ul : 10 * 1000000000ul;\n" \
"\n" \
"	// u is a primitive (2^e)th root of unity\n" \
"	uint32 e = 1; for (uint64 k = (p - 1) / 2; k % 2 == 0; k /= 2) ++e;\n" \
"	e = min(e, (uint32)(g_n - 1));\n" \
"\n" \
"	uint32 d = 2; uint64 md = add_mod(one, one, p);\n" \
"	while (d < 256)\n" \
"	{\n" \
"		if (jacobi(d, p) == -1) break;\n" \
"		++d; md = add_mod(md, one, p); if ((d == 4) || (d == 9)) { ++d; md = add_mod(md, one, p); }\n" \
"	}\n" \
"	if (d >= 256) return;\n" \
"\n" \
"	const uint64 u = pow_mod(md, (p - 1) >> e, p, q);\n" \
"\n" \
"	if (r1 != 0)\n" \
"	{\n" \
"		uint64 b = toInt(r1, p, q);\n" \
"		for (uint32 i = 0, s = 1u << e; i < s; i += 2)\n" \
"		{\n" \
"			push_factor(b, bMax, p, factor_count, factor);\n" \
"			push_factor(p - b, bMax, p, factor_count, factor);\n" \
"			b = mul_mod(b, u, p, q);\n" \
"		}\n" \
"	}\n" \
"\n" \
"	if (r2 != 0)\n" \
"	{\n" \
"		uint64 b = toInt(r2, p, q);\n" \
"		for (uint32 i = 0, s = 1u << e; i < s; i += 2)\n" \
"		{\n" \
"			push_factor(b, bMax, p, factor_count, factor);\n" \
"			push_factor(p - b, bMax, p, factor_count, factor);\n" \
"			b = mul_mod(b, u, p, q);\n" \
"		}\n" \
"	}\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void clear_primes_neg(__global uint * restrict const prime_count, __global uint * restrict const _prime_3_4_count, __global uint * restrict const _prime_5_8_count)\n" \
"{\n" \
"	*prime_count = 0;\n" \
"	*_prime_3_4_count = 0;\n" \
"	*_prime_5_8_count = 0;\n" \
"}\n" \
"";
