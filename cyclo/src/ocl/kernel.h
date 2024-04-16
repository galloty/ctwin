/*
Copyright 2023, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <cstdint>

static const char * const src_ocl_kernel = \
"/*\n" \
"Copyright 2023, Yves Gallot\n" \
"\n" \
"ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\n" \
"Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n" \
"*/\n" \
"\n" \
"#define P1P2	(P1 * (uint64)P2)\n" \
"#define P2P3	(P2 * (uint64)P3)\n" \
"\n" \
"typedef uint	sz_t;\n" \
"\n" \
"typedef uint	uint32;\n" \
"typedef int 	int32;\n" \
"typedef ulong	uint64;\n" \
"typedef long 	int64;\n" \
"typedef uint2	uint32_2;\n" \
"\n" \
"typedef struct { uint32 r1; uint32 r2; uint32 r3; } uint32_3;\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"typedef struct { uint64 s0; uint32 s1; } uint96;\n" \
"typedef struct { uint64 s0; int32  s1; } int96;\n" \
"\n" \
"inline int96 int96_zero() { int96 r; r.s0 = 0; r.s1 = 0; return r; }\n" \
"inline int96 int96_set_si(const int64 n) { int96 r; r.s0 = (uint64)n; r.s1 = (n < 0) ? -1 : 0; return r; }\n" \
"\n" \
"inline uint96 uint96_set(const uint64 s0, const uint32 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }\n" \
"\n" \
"inline int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int32)x.s1; return r; }\n" \
"inline uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = (uint32)x.s1; return r; }\n" \
"\n" \
"inline bool int96_is_neg(const int96 x) { return (x.s1 < 0); }\n" \
"\n" \
"inline bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }\n" \
"\n" \
"inline int96 int96_neg(const int96 x)\n" \
"{\n" \
"	const int32 c = (x.s0 != 0) ? 1 : 0;\n" \
"	int96 r; r.s0 = -x.s0; r.s1 = -x.s1 - c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline int96 int96_add(const int96 x, const int96 y)\n" \
"{\n" \
"	const uint64 s0 = x.s0 + y.s0;\n" \
"	const int32 c = (s0 < y.s0) ? 1 : 0;\n" \
"	int96 r; r.s0 = s0; r.s1 = x.s1 + y.s1 + c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint96 uint96_add_64(const uint96 x, const uint64 y)\n" \
"{\n" \
"	const uint64 s0 = x.s0 + y;\n" \
"	const uint32 c = (s0 < y) ? 1 : 0;\n" \
"	uint96 r; r.s0 = s0; r.s1 = x.s1 + c;\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline int96 uint96_subi(const uint96 x, const uint96 y)\n" \
"{\n" \
"	const uint32 c = (x.s0 < y.s0) ? 1 : 0;\n" \
"	int96 r; r.s0 = x.s0 - y.s0; r.s1 = (int32)(x.s1 - y.s1 - c);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint96 uint96_mul_64_32(const uint64 x, const uint32 y)\n" \
"{\n" \
"	const uint32 x_lo = (uint32)x, x_hi = (uint32)(x >> 32);\n" \
"	const uint32 l_lo = x_lo * y, l_hi = mul_hi(x_lo, y);\n" \
"	const uint64 h = x_hi * (uint64)y + l_hi;\n" \
"	uint96 r; r.s0 = (h << 32) | l_lo; r.s1 = (uint32)(h >> 32);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint96 int96_abs(const int96 x)\n" \
"{\n" \
"	const int96 t = (int96_is_neg(x)) ? int96_neg(x) : x;\n" \
"	return int96_u(t);\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline uint32 _addMod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"	const uint32 c = (lhs >= p - rhs) ? p : 0;\n" \
"	return lhs + rhs - c;\n" \
"}\n" \
"\n" \
"inline uint32 _subMod(const uint32 lhs, const uint32 rhs, const uint32 p)\n" \
"{\n" \
"	const uint32 c = (lhs < rhs) ? p : 0;\n" \
"	return lhs - rhs + c;\n" \
"}\n" \
"\n" \
"inline uint32 _halfMod(const uint32 lhs, const uint32 p)\n" \
"{\n" \
"	return (lhs % 2 == 0) ? lhs / 2 : hadd(lhs, p);\n" \
"}\n" \
"\n" \
"// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.\n" \
"// Montgomery form (lhs, rhs and output): if 0 <= r < p then f is r * 2^32 mod p\n" \
"inline uint32 _mulMonty(const uint32 lhs, const uint32 rhs, const uint32 p, const uint32 q)\n" \
"{\n" \
"	const uint32 t_lo = lhs * rhs, t_hi = mul_hi(lhs, rhs);\n" \
"	const uint32 mp = mul_hi(t_lo * q, p);\n" \
"	return _subMod(t_hi, mp, p);\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline uint32 add_P1(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P1); }\n" \
"inline uint32 add_P2(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P2); }\n" \
"inline uint32 add_P3(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P3); }\n" \
"\n" \
"inline uint32 sub_P1(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P1); }\n" \
"inline uint32 sub_P2(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P2); }\n" \
"inline uint32 sub_P3(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P3); }\n" \
"\n" \
"inline uint32 half_P1(const uint32 lhs) { return _halfMod(lhs, P1); }\n" \
"inline uint32 half_P2(const uint32 lhs) { return _halfMod(lhs, P2); }\n" \
"inline uint32 half_P3(const uint32 lhs) { return _halfMod(lhs, P3); }\n" \
"\n" \
"inline uint32 mul_P1(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P1, Q1); }\n" \
"inline uint32 mul_P2(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P2, Q2); }\n" \
"inline uint32 mul_P3(const uint32 lhs, const uint32 rhs) { return _mulMonty(lhs, rhs, P3, Q3); }\n" \
"\n" \
"inline uint32 toMonty_P1(const uint32 lhs) { return _mulMonty(lhs, R1, P1, Q1); }\n" \
"inline uint32 toMonty_P2(const uint32 lhs) { return _mulMonty(lhs, R2, P2, Q2); }\n" \
"inline uint32 toMonty_P3(const uint32 lhs) { return _mulMonty(lhs, R3, P3, Q3); }\n" \
"\n" \
"inline uint32 seti_P1(const int32 i) { return (i < 0) ? (uint32)(i + P1) : (uint32)i; }\n" \
"inline uint32 seti_P2(const int32 i) { return (i < 0) ? (uint32)(i + P2) : (uint32)i; }\n" \
"inline uint32 seti_P3(const int32 i) { return (i < 0) ? (uint32)(i + P3) : (uint32)i; }\n" \
"\n" \
"inline int32 geti_P1(const uint32 n) { return (n > P1 / 2) ? (int32)(n - P1) : (int32)n; }\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline uint32_3 add_3(const uint32_3 lhs, const uint32_3 rhs) { const uint32_3 r = { add_P1(lhs.r1, rhs.r1), add_P2(lhs.r2, rhs.r2), add_P3(lhs.r3, rhs.r3) }; return r; }\n" \
"inline uint32_3 sub_3(const uint32_3 lhs, const uint32_3 rhs) { const uint32_3 r = { sub_P1(lhs.r1, rhs.r1), sub_P2(lhs.r2, rhs.r2), sub_P3(lhs.r3, rhs.r3) }; return r; }\n" \
"inline uint32_3 half_3(const uint32_3 lhs) { const uint32_3 r = { half_P1(lhs.r1), half_P2(lhs.r2), half_P3(lhs.r3) }; return r; }\n" \
"inline uint32_3 mul_3(const uint32_3 lhs, const uint32_3 rhs) { const uint32_3 r = { mul_P1(lhs.r1, rhs.r1), mul_P2(lhs.r2, rhs.r2), mul_P3(lhs.r3, rhs.r3) }; return r; }\n" \
"inline uint32_3 toMonty_3(const uint32_3 lhs) { const uint32_3 r = { toMonty_P1(lhs.r1), toMonty_P2(lhs.r2), toMonty_P3(lhs.r3) }; return r; }\n" \
"inline uint32_3 seti_3(const int32 i) { const uint32_3 r = { seti_P1(i), seti_P2(i), seti_P3(i) }; return r; }\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"__constant uint64 mask64[64] = {\n" \
"	0x0000000000000001ul, 0x0000000000000002ul, 0x0000000000000004ul, 0x0000000000000008ul,\n" \
"	0x0000000000000010ul, 0x0000000000000020ul, 0x0000000000000040ul, 0x0000000000000080ul,\n" \
"	0x0000000000000100ul, 0x0000000000000200ul, 0x0000000000000400ul, 0x0000000000000800ul,\n" \
"	0x0000000000001000ul, 0x0000000000002000ul, 0x0000000000004000ul, 0x0000000000008000ul,\n" \
"	0x0000000000010000ul, 0x0000000000020000ul, 0x0000000000040000ul, 0x0000000000080000ul,\n" \
"	0x0000000000100000ul, 0x0000000000200000ul, 0x0000000000400000ul, 0x0000000000800000ul,\n" \
"	0x0000000001000000ul, 0x0000000002000000ul, 0x0000000004000000ul, 0x0000000008000000ul,\n" \
"	0x0000000010000000ul, 0x0000000020000000ul, 0x0000000040000000ul, 0x0000000080000000ul,\n" \
"	0x0000000100000000ul, 0x0000000200000000ul, 0x0000000400000000ul, 0x0000000800000000ul,\n" \
"	0x0000001000000000ul, 0x0000002000000000ul, 0x0000004000000000ul, 0x0000008000000000ul,\n" \
"	0x0000010000000000ul, 0x0000020000000000ul, 0x0000040000000000ul, 0x0000080000000000ul,\n" \
"	0x0000100000000000ul, 0x0000200000000000ul, 0x0000400000000000ul, 0x0000800000000000ul,\n" \
"	0x0001000000000000ul, 0x0002000000000000ul, 0x0004000000000000ul, 0x0008000000000000ul,\n" \
"	0x0010000000000000ul, 0x0020000000000000ul, 0x0040000000000000ul, 0x0080000000000000ul,\n" \
"	0x0100000000000000ul, 0x0200000000000000ul, 0x0400000000000000ul, 0x0800000000000000ul,\n" \
"	0x1000000000000000ul, 0x2000000000000000ul, 0x4000000000000000ul, 0x8000000000000000ul };\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline uint32_3 read1(const __global uint32_2 * const x12, const __global uint32 * const x3, const sz_t k)\n" \
"{\n" \
"	const uint32_2 t = x12[k]; const uint32_3 r = { t.s0, t.s1, x3[k] }; return r;\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline void frwd2(uint32_3 * const u, const uint32_3 w)\n" \
"{\n" \
"	const uint32_3 u1w = mul_3(u[1], w);\n" \
"	u[1] = sub_3(u[0], u1w); u[0] = add_3(u[0], u1w);\n" \
"}\n" \
"\n" \
"inline void bkwd2(uint32_3 * const u, const uint32_3 wi)\n" \
"{\n" \
"	const uint32_3 v1 = sub_3(u[0], u[1]);\n" \
"	u[0] = add_3(u[0], u[1]); u[1] = mul_3(v1, wi);\n" \
"}\n" \
"\n" \
"inline void sqr2(uint32_3 * const u, const uint32_3 w)\n" \
"{\n" \
"	const uint32_3 u1w = mul_3(u[1], w);\n" \
"	u[1] = mul_3(add_3(u[0], u[0]), u[1]);\n" \
"	u[0] = add_3(mul_3(u[0], u[0]), mul_3(u1w, u1w));\n" \
"}\n" \
"\n" \
"inline void read2(uint32_3 * const u, const __global uint32_2 * const x12, const __global uint32 * const x3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) { const sz_t j = k + h * VSIZE * m; const uint32_2 t = x12[j]; const uint32_3 r = { t.s0, t.s1, x3[j] }; u[h] = r; }\n" \
"}\n" \
"\n" \
"inline void write2(__global uint32_2 * const x12, __global uint32 * const x3, const uint32_3 * const u, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) { const sz_t j = k + h * VSIZE * m; x12[j] = (uint32_2)(u[h].r1, u[h].r2); x3[j] = u[h].r3; }\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline void frwd41(uint32_3 * const u, const uint32_3 w_1)\n" \
"{\n" \
"	for (sz_t i = 0; i < 2; ++i)\n" \
"	{\n" \
"		const uint32_3 uhw = mul_3(u[i + 2], w_1);\n" \
"		u[i + 2] = sub_3(u[i + 0], uhw); u[i + 0] = add_3(u[i + 0], uhw);\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void frwd42(uint32_3 * const u, const uint32_3 w_23[2])\n" \
"{\n" \
"	for (sz_t j = 0; j < 2; ++j)\n" \
"	{\n" \
"		const uint32_3 uhw = mul_3(u[2 * j + 1], w_23[j]);\n" \
"		u[2 * j + 1] = sub_3(u[2 * j + 0], uhw); u[2 * j + 0] = add_3(u[2 * j + 0], uhw);\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void bkwd42(uint32_3 * const u, const uint32_3 wi_23[2])\n" \
"{\n" \
"	for (sz_t j = 0; j < 2; ++j)\n" \
"	{\n" \
"		const uint32_3 vh = mul_3(sub_3(u[2 * j + 0], u[2 * j + 1]), wi_23[j]);\n" \
"		u[2 * j + 0] = add_3(u[2 * j + 0], u[2 * j + 1]); u[2 * j + 1] = vh;\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void bkwd41(uint32_3 * const u, const uint32_3 wi_1)\n" \
"{\n" \
"	for (sz_t i = 0; i < 2; ++i)\n" \
"	{\n" \
"		const uint32_3 vh = mul_3(sub_3(u[i + 0], u[i + 2]), wi_1);\n" \
"		u[i + 0] = add_3(u[i + 0], u[i + 2]); u[i + 2] = vh;\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void frwd41_0(uint32_3 * const u, const uint32_3 w_1)\n" \
"{\n" \
"	for (sz_t i = 0; i < 2; ++i)\n" \
"	{\n" \
"		const uint32_3 uhw = mul_3(u[i + 2], w_1);\n" \
"		u[i + 2] = add_3(u[i + 2], sub_3(u[i + 0], uhw)); u[i + 0] = add_3(u[i + 0], uhw);\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void frwd42_0(uint32_3 * const u, const uint32_3 w_23[2])\n" \
"{\n" \
"	for (sz_t j = 0; j < 2; ++j)\n" \
"	{\n" \
"		const uint32_3 ul = toMonty_3(u[2 * j + 0]), uhw = mul_3(u[2 * j + 1], w_23[j]);\n" \
"		u[2 * j + 0] = add_3(ul, uhw); u[2 * j + 1] = sub_3(ul, uhw);\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void bkwd41_0(uint32_3 * const u, const uint32_3 wi_0, const uint32_3 wi_1)\n" \
"{\n" \
"	for (sz_t i = 0; i < 2; ++i)\n" \
"	{\n" \
"		const uint32_3 vh = mul_3(add_3(u[i + 0], u[i + 2]), wi_0);\n" \
"		u[i + 2] = mul_3(sub_3(u[i + 2], u[i + 0]), wi_1); u[i + 0] = mul_3(sub_3(u[i + 0], vh), wi_1);\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void sqr42(uint32_3 * const u, const uint32_3 w_23[2])\n" \
"{\n" \
"	for (sz_t j = 0; j < 2; ++j)\n" \
"	{\n" \
"		const uint32_3 uhw = mul_3(u[2 * j + 1], w_23[j]);\n" \
"		u[2 * j + 1] = mul_3(add_3(u[2 * j + 0], u[2 * j + 0]), u[2 * j + 1]); u[2 * j + 0] = add_3(mul_3(u[2 * j + 0], u[2 * j + 0]), mul_3(uhw, uhw));\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void read4(uint32_3 * const u, const __global uint32_2 * const x12, const __global uint32 * const x3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + h * VSIZE * m; const uint32_2 t = x12[j]; const uint32_3 r = { t.s0, t.s1, x3[j] }; u[h] = r; }\n" \
"}\n" \
"\n" \
"inline void write4(__global uint32_2 * const x12, __global uint32 * const x3, const uint32_3 * const u, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + h * VSIZE * m; x12[j] = (uint32_2)(u[h].r1, u[h].r2); x3[j] = u[h].r3; }\n" \
"}\n" \
"\n" \
"inline void read4_2(uint32_3 * const u, const __global uint32_2 * const x12, const __global uint32 * const x3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + (h / 2) * VSIZE * m + (h % 2) * (VSIZE / 2); const uint32_2 t = x12[j]; const uint32_3 r = { t.s0, t.s1, x3[j] }; u[h] = r; }\n" \
"}\n" \
"\n" \
"inline void write4_2(__global uint32_2 * const x12, __global uint32 * const x3, const uint32_3 * const u, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + (h / 2) * VSIZE * m + (h % 2) * (VSIZE / 2); x12[j] = (uint32_2)(u[h].r1, u[h].r2); x3[j] = u[h].r3; }\n" \
"}\n" \
"\n" \
"inline void read22l(uint32_3 * const u, const __local uint32_2 * const X12, const __local uint32 * const X3, const sz_t k)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + (h / 2) + (h % 2) * VSIZE; const uint32_2 t = X12[j]; const uint32_3 r = { t.s0, t.s1, X3[j] }; u[h] = r; }\n" \
"}\n" \
"\n" \
"inline void write22l(__local uint32_2 * const X12, __local uint32 * const X3, const uint32_3 * const u, const sz_t k)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + (h / 2) + (h % 2) * VSIZE; X12[j] = (uint32_2)(u[h].r1, u[h].r2); X3[j] = u[h].r3; }\n" \
"}\n" \
"\n" \
"inline void read4l(uint32_3 * const u, const __local uint32_2 * const X12, const __local uint32 * const X3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + h * VSIZE * m; const uint32_2 t = X12[j]; const uint32_3 r = { t.s0, t.s1, X3[j] }; u[h] = r; }\n" \
"}\n" \
"\n" \
"inline void write4l(__local uint32_2 * const X12, __local uint32 * const X3, const uint32_3 * const u, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + h * VSIZE * m; X12[j] = (uint32_2)(u[h].r1, u[h].r2); X3[j] = u[h].r3; }\n" \
"}\n" \
"\n" \
"inline void read4l_2(uint32_3 * const u, const __local uint32_2 * const X12, const __local uint32 * const X3, const sz_t k)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + h * (VSIZE / 2); const uint32_2 t = X12[j]; const uint32_3 r = { t.s0, t.s1, X3[j] }; u[h] = r; }\n" \
"}\n" \
"\n" \
"inline void write4l_2(__local uint32_2 * const X12, __local uint32 * const X3, const uint32_3 * const u, const sz_t k)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) { const sz_t j = k + h * (VSIZE / 2); X12[j] = (uint32_2)(u[h].r1, u[h].r2); X3[j] = u[h].r3; }\n" \
"}\n" \
"\n" \
"inline void frwd4(uint32_3 * const u, const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3, const sz_t sj)\n" \
"{\n" \
"	frwd41(u, read1(wr12, wr3, sj));\n" \
"	uint32_3 w_23[2]; w_23[0] = read1(wr12, wr3, 2 * sj); w_23[1] = read1(wr12, wr3, 2 * sj + 1); \n" \
"	frwd42(u, w_23);\n" \
"}\n" \
"\n" \
"inline void bkwd4(uint32_3 * const u, const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3, const sz_t sj)\n" \
"{\n" \
"	uint32_3 wi_23[2]; wi_23[0] = read1(wri12, wri3, 2 * sj); wi_23[1] = read1(wri12, wri3, 2 * sj + 1); \n" \
"	bkwd42(u, wi_23);\n" \
"	bkwd41(u, read1(wri12, wri3, sj));\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline void frwd81(uint32_3 * const u, const uint32_3 w_1)\n" \
"{\n" \
"	for (sz_t i = 0; i < 4; ++i)\n" \
"	{\n" \
"		const uint32_3 uhw = mul_3(u[i + 4], w_1);\n" \
"		u[i + 4] = sub_3(u[i + 0], uhw); u[i + 0] = add_3(u[i + 0], uhw);\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void frwd82(uint32_3 * const u, const uint32_3 w_2, const uint32_3 w_3)\n" \
"{\n" \
"	for (sz_t i = 0; i < 2; ++i)\n" \
"	{\n" \
"		const uint32_3 uhw = mul_3(u[i + 2], w_2);\n" \
"		u[i + 2] = sub_3(u[i + 0], uhw); u[i + 0] = add_3(u[i + 0], uhw);\n" \
"	}\n" \
"	for (sz_t i = 0; i < 2; ++i)\n" \
"	{\n" \
"		const uint32_3 uhw = mul_3(u[i + 6], w_3);\n" \
"		u[i + 6] = sub_3(u[i + 4], uhw); u[i + 4] = add_3(u[i + 4], uhw);\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void frwd83(uint32_3 * const u, const uint32_3 w_4, const uint32_3 w_5, const uint32_3 w_6, const uint32_3 w_7)\n" \
"{\n" \
"	const uint32_3 u1w4 = mul_3(u[1], w_4);\n" \
"	u[1] = sub_3(u[0], u1w4); u[0] = add_3(u[0], u1w4);\n" \
"	const uint32_3 u3w5 = mul_3(u[3], w_5);\n" \
"	u[3] = sub_3(u[2], u3w5); u[2] = add_3(u[2], u3w5);\n" \
"	const uint32_3 u5w6 = mul_3(u[5], w_6);\n" \
"	u[5] = sub_3(u[4], u5w6); u[4] = add_3(u[4], u5w6);\n" \
"	const uint32_3 u7w7 = mul_3(u[7], w_7);\n" \
"	u[7] = sub_3(u[6], u7w7); u[6] = add_3(u[6], u7w7);\n" \
"}\n" \
"\n" \
"inline void bkwd83(uint32_3 * const u, const uint32_3 wi_4, const uint32_3 wi_5, const uint32_3 wi_6, const uint32_3 wi_7)\n" \
"{\n" \
"	const uint32_3 v1 = sub_3(u[0], u[1]);\n" \
"	u[0] = add_3(u[0], u[1]); u[1] = mul_3(v1, wi_4);\n" \
"	const uint32_3 v3 = sub_3(u[2], u[3]);\n" \
"	u[2] = add_3(u[2], u[3]); u[3] = mul_3(v3, wi_5);\n" \
"	const uint32_3 v5 = sub_3(u[4], u[5]);\n" \
"	u[4] = add_3(u[4], u[5]); u[5] = mul_3(v5, wi_6);\n" \
"	const uint32_3 v7 = sub_3(u[6], u[7]);\n" \
"	u[6] = add_3(u[6], u[7]); u[7] = mul_3(v7, wi_7);\n" \
"}\n" \
"\n" \
"inline void bkwd82(uint32_3 * const u, const uint32_3 wi_2, const uint32_3 wi_3)\n" \
"{\n" \
"	const uint32_3 v2 = sub_3(u[0], u[2]);\n" \
"	u[0] = add_3(u[0], u[2]); u[2] = mul_3(v2, wi_2);\n" \
"	const uint32_3 v3 = sub_3(u[1], u[3]);\n" \
"	u[1] = add_3(u[1], u[3]); u[3] = mul_3(v3, wi_2);\n" \
"\n" \
"	const uint32_3 v6 = sub_3(u[4], u[6]);\n" \
"	u[4] = add_3(u[4], u[6]); u[6] = mul_3(v6, wi_3);\n" \
"	const uint32_3 v7 = sub_3(u[5], u[7]);\n" \
"	u[5] = add_3(u[5], u[7]); u[7] = mul_3(v7, wi_3);\n" \
"}\n" \
"\n" \
"inline void bkwd81(uint32_3 * const u, const uint32_3 wi_1)\n" \
"{\n" \
"	for (sz_t i = 0; i < 4; ++i)\n" \
"	{\n" \
"		const uint32_3 vh = sub_3(u[i + 0], u[i + 4]);\n" \
"		u[i + 0] = add_3(u[i + 0], u[i + 4]); u[i + 4] = mul_3(vh, wi_1);\n" \
"	}\n" \
"}\n" \
"\n" \
"inline void read8(uint32_3 * const u, const __global uint32_2 * const x12, const __global uint32 * const x3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 8; ++h) { const uint32_2 t = x12[k + h * VSIZE * m]; const uint32_3 r = { t.s0, t.s1, x3[k + h * VSIZE * m] }; u[h] = r; }\n" \
"}\n" \
"\n" \
"inline void write8(__global uint32_2 * const x12, __global uint32 * const x3, const uint32_3 * const u, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 8; ++h) { x12[k + h * VSIZE * m] = (uint32_2)(u[h].r1, u[h].r2); x3[k + h * VSIZE * m] = u[h].r3; }\n" \
"}\n" \
"\n" \
"inline void read8l(uint32_3 * const u, const __local uint32_2 * const X12, const __local uint32 * const X3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 8; ++h) { const uint32_2 t = X12[k + h * VSIZE * m]; const uint32_3 r = { t.s0, t.s1, X3[k + h * VSIZE * m] }; u[h] = r; }\n" \
"}\n" \
"\n" \
"inline void write8l(__local uint32_2 * const X12, __local uint32 * const X3, const uint32_3 * const u, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 8; ++h) { X12[k + h * VSIZE * m] = (uint32_2)(u[h].r1, u[h].r2); X3[k + h * VSIZE * m] = u[h].r3; }\n" \
"}\n" \
"\n" \
"inline void frwd8(uint32_3 * const u, const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3, const sz_t sj)\n" \
"{\n" \
"	frwd81(u, read1(wr12, wr3, sj));\n" \
"	frwd82(u, read1(wr12, wr3, 2 * sj), read1(wr12, wr3, 2 * sj + 1));\n" \
"	frwd83(u, read1(wr12, wr3, 4 * sj), read1(wr12, wr3, 4 * sj + 1), read1(wr12, wr3, 4 * sj + 2), read1(wr12, wr3, 4 * sj + 3));\n" \
"}\n" \
"\n" \
"inline void bkwd8(uint32_3 * const u, const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3, const sz_t sj)\n" \
"{\n" \
"	bkwd83(u, read1(wri12, wri3, 4 * sj), read1(wri12, wri3, 4 * sj + 1), read1(wri12, wri3, 4 * sj + 2), read1(wri12, wri3, 4 * sj + 3));\n" \
"	bkwd82(u, read1(wri12, wri3, 2 * sj), read1(wri12, wri3, 2 * sj + 1));\n" \
"	bkwd81(u, read1(wri12, wri3, sj));\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"inline static int96 garner3(const uint32_3 lhs)\n" \
"{\n" \
"	const uint32 u13 = mul_P1(sub_P1(lhs.r1, lhs.r3), InvP3_P1);\n" \
"	const uint32 u23 = mul_P2(sub_P2(lhs.r2, lhs.r3), InvP3_P2);\n" \
"	const uint32 u123 = mul_P1(sub_P1(u13, u23), InvP2_P1);\n" \
"	const uint96 n = uint96_add_64(uint96_mul_64_32(P2P3, u123), u23 * (uint64)P3 + lhs.r3);\n" \
"	const uint96 P1P2P3 = uint96_set(P1P2P3l, P1P2P3h), P1P2P3_2 = uint96_set(P1P2P3_2l, P1P2P3_2h);\n" \
"	const int96 r = uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint32 barrett(const uint64 a, const uint32 b, const uint32 b_inv, const int b_s, uint32 * a_p)\n" \
"{\n" \
"	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.\n" \
"	// b < 2^31, alpha = 2^29 => a < 2^29 b\n" \
"	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31\n" \
"	// b_inv = [2^{s + 32} / b]\n" \
"	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31\n" \
"	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].\n" \
"	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])\n" \
"	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,\n" \
"	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.\n" \
"\n" \
"	const uint32 d = mul_hi((uint32)(a >> b_s), b_inv), r = (uint32)a - d * b;\n" \
"	const bool o = (r >= b);\n" \
"	*a_p = o ? d + 1 : d;\n" \
"	return o ? r - b : r;\n" \
"}\n" \
"\n" \
"inline int32 reduce64(int64 * f, const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32\n" \
"	// 2- t < 2^22 b^2 => t_h < b^2 / 2^7. If 2 <= b < 32 then t_h < 32^2 / 2^7 = 2^8 < 2^29 b\n" \
"	const uint64 t = abs(*f);\n" \
"	const uint64 t_h = t >> 29;\n" \
"	const uint32 t_l = (uint32)t & ((1u << 29) - 1);\n" \
"\n" \
"	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint32 d_l, r_l = barrett(((uint64)r_h << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const uint64 d = ((uint64)d_h << 29) | d_l;\n" \
"\n" \
"	const bool s = (*f < 0);\n" \
"	*f = s ? -(int64)d : (int64)d;\n" \
"	return s ? -(int32)r_l : (int32)r_l;\n" \
"}\n" \
"\n" \
"inline int32 reduce96(int96 * f, const uint32 b, const uint32 b_inv, const int b_s)\n" \
"{\n" \
"	const uint96 t = int96_abs(*f);\n" \
"	const uint64 t_h = ((uint64)t.s1 << (64 - 29)) | (t.s0 >> 29);\n" \
"	const uint32 t_l = (uint32)t.s0 & ((1u << 29) - 1);\n" \
"\n" \
"	uint32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);\n" \
"	uint32 d_l, r_l = barrett(((uint64)r_h << 29) | t_l, b, b_inv, b_s, &d_l);\n" \
"	const uint64 d = ((uint64)d_h << 29) | d_l;\n" \
"\n" \
"	const bool s = int96_is_neg(*f);\n" \
"	*f = int96_set_si(s ? -(int64)d : (int64)d);\n" \
"	return s ? -(int32)r_l : (int32)r_l;\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"\n" \
"__kernel\n" \
"void set(__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 a)\n" \
"{\n" \
"	const sz_t k = (sz_t)get_global_id(0);\n" \
"	const uint32 v = (k < VSIZE) ? a : 0;\n" \
"	x12[k] = (uint32_2)(v, v); x3[k] = v;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void copy(__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 reg_dst, const uint32 reg_src)\n" \
"{\n" \
"	const sz_t k = (sz_t)get_global_id(0);\n" \
"	x12[reg_dst * VSIZE * NSIZE + k] = x12[reg_src * VSIZE * NSIZE + k];\n" \
"	x3[reg_dst * VSIZE * NSIZE + k] = x3[reg_src * VSIZE * NSIZE + k];\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square2(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	// get_global_size(0) is VSIZE * NSIZE / 2\n" \
"	const sz_t sj = NSIZE / 2 + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_3 u[2]; read2(u, x12, x3, k, 1);\n" \
"	sqr2(u, read1(wr12, wr3, sj));\n" \
"	write2(x12, x3, u, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	// get_global_size(0) is VSIZE * NSIZE / 4\n" \
"	const sz_t sj = NSIZE / 4 + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_3 u[4]; read4(u, x12, x3, k, 1);\n" \
"	frwd41(u, read1(wr12, wr3, sj));\n" \
"	uint32_3 w_23[2]; w_23[0] = read1(wr12, wr3, 2 * sj); w_23[1] = read1(wr12, wr3, 2 * sj + 1); \n" \
"	sqr42(u, w_23);\n" \
"	bkwd41(u, read1(wri12, wri3, sj));\n" \
"	write4(x12, x3, u, k, 1);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(8 / 4 * VSIZE, 1, 1)))\n" \
"void square8(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32_2 X12[8 * VSIZE];\n" \
"	__local uint32 X3[8 * VSIZE];\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"\n" \
"	const sz_t k_group = group_id * 8 * VSIZE;\n" \
"\n" \
"	// get_global_size(0) is VSIZE * NSIZE / 4\n" \
"	// const sz_t global_id = 8 / 4 * VSIZE * group_id + i, sj8 = NSIZE / 2 + global_id * 2 / VSIZE;\n" \
"	const sz_t sj8 = NSIZE / 2 + 8 / 4 * 2 * group_id + i * 2 / VSIZE, sj2 = sj8 / 4;\n" \
"	const sz_t k2 = 2 * ((2 * i) & (sz_t)~(VSIZE - 1)) + ((2 * i) % VSIZE), k8 = i;\n" \
"\n" \
"	uint32_3 u[4];\n" \
"	\n" \
"	read4(u, x12, x3, k_group + k8, 2);\n" \
"	frwd4(u, wr12, wr3, sj2);\n" \
"	write4l(X12, X3, u, k8, 2);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read22l(u, X12, X3, k2);\n" \
"	const uint32_3 w = read1(wr12, wr3, sj8);\n" \
"	sqr2(&u[0], w); sqr2(&u[2], w);\n" \
"	write22l(X12, X3, u, k2);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, k8, 2);\n" \
"	bkwd4(u, wri12, wri3, sj2);\n" \
"	write4(x12, x3, u, k_group + k8, 2);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16 / 4 * VSIZE, 1, 1)))\n" \
"void square16(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"	const sz_t il = i % VSIZE, iv = i & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	const sz_t k_group = group_id * 16 * VSIZE;\n" \
"\n" \
"	// get_global_size(0) is VSIZE * NSIZE / 4\n" \
"	// const sz_t global_id = 16 / 4 * VSIZE * group_id + i, sj4 = NSIZE / 4 + global_id / VSIZE;\n" \
"	const sz_t sj4 = NSIZE / 4 + 16 / 4 * group_id + i / VSIZE, sj = sj4 / 4;\n" \
"	const sz_t k4 = 4 * iv + il, k16 = i;	// iv + il\n" \
"\n" \
"	uint32_3 u[4];\n" \
"\n" \
"	read4(u, x12, x3, k_group + k16, 4);\n" \
"	frwd4(u, wr12, wr3, sj);\n" \
"	write4l(X12, X3, u, k16, 4);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, k4, 1);\n" \
"	frwd41(u, read1(wr12, wr3, sj4));\n" \
"	uint32_3 w_23[2]; w_23[0] = read1(wr12, wr3, 2 * sj4); w_23[1] = read1(wr12, wr3, 2 * sj4 + 1); \n" \
"	sqr42(u, w_23);\n" \
"	bkwd41(u, read1(wri12, wri3, sj4));\n" \
"	write4l(X12, X3, u, k4, 1);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, k16, 4);\n" \
"	bkwd4(u, wri12, wri3, sj);\n" \
"	write4(x12, x3, u, k_group + k16, 4);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(32 / 4 * VSIZE, 1, 1)))\n" \
"void square32(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32_2 X12[32 * VSIZE];	// VSIZE = 64 => 16 KB\n" \
"	__local uint32 X3[32 * VSIZE];\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"\n" \
"	const sz_t k_group = group_id * 32 * VSIZE;\n" \
"\n" \
"	// get_global_size(0) is VSIZE * NSIZE / 4\n" \
"	// const sz_t global_id = 32 / 4 * VSIZE * group_id + i, sj32 = NSIZE / 2 + global_id * 2 / VSIZE;\n" \
"	const sz_t sj32 = NSIZE / 2 + 32 / 4 * 2 * group_id + i / (VSIZE / 2), sj8 = sj32 / 4, sj2 = sj8 / 4;\n" \
"	const sz_t k2 = 2 * ((2 * i) & (sz_t)~(VSIZE - 1)) + ((2 * i) % VSIZE);\n" \
"	const sz_t k8 = 4 * (i & (sz_t)~(2 * VSIZE - 1)) + (i % (2 * VSIZE));\n" \
"	const sz_t k32 = i;\n" \
"\n" \
"	uint32_3 u[4];\n" \
"\n" \
"	read4(u, x12, x3, k_group + k32, 8);\n" \
"	frwd4(u, wr12, wr3, sj2);\n" \
"	write4l(X12, X3, u, k32, 8);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, k8, 2);\n" \
"	frwd4(u, wr12, wr3, sj8);\n" \
"	write4l(X12, X3, u, k8, 2);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read22l(u, X12, X3, k2);\n" \
"	const uint32_3 w = read1(wr12, wr3, sj32);\n" \
"	sqr2(&u[0], w); sqr2(&u[2], w);\n" \
"	write22l(X12, X3, u, k2);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, k8, 2);\n" \
"	bkwd4(u, wri12, wri3, sj8);\n" \
"	write4l(X12, X3, u, k8, 2);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, k32, 8);\n" \
"	bkwd4(u, wri12, wri3, sj2);\n" \
"	write4(x12, x3, u, k_group + k32, 8);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12,  __global uint32 * restrict const x3)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	// get_global_size(0) is VSIZE * NSIZE / 2\n" \
"	const sz_t sj = NSIZE / 2 + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_3 u[2], v[2]; read2(u, x12, x3, k, 1); read2(v, x12, x3, VSIZE * NSIZE + k, 1);\n" \
"	const uint32_3 w_1 = read1(wr12, wr3, sj);\n" \
"	frwd2(u, w_1); frwd2(v, w_1);\n" \
"\n" \
"	for (sz_t h = 0; h < 2; ++h) u[h] = half_3(mul_3(u[h], v[h]));\n" \
"\n" \
"	bkwd2(u, read1(wri12, wri3, sj));\n" \
"	write2(x12, x3, u, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	// get_global_size(0) is VSIZE * NSIZE / 4\n" \
"	const sz_t sj = NSIZE / 4 + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_3 u[4], v[4]; read4(u, x12, x3, k, 1); read4(v, x12, x3, VSIZE * NSIZE + k, 1);\n" \
"	const uint32_3 w_1 = read1(wr12, wr3, sj);\n" \
"	frwd41(u, w_1); frwd41(v, w_1);\n" \
"	uint32_3 w_23[2]; w_23[0] = read1(wr12, wr3, 2 * sj); w_23[1] = read1(wr12, wr3, 2 * sj + 1); \n" \
"	frwd42(u, w_23); frwd42(v, w_23);\n" \
"\n" \
"	for (sz_t h = 0; h < 4; ++h) u[h] = half_3(mul_3(u[h], v[h]));\n" \
"\n" \
"	bkwd4(u, wri12, wri3, sj);\n" \
"	write4(x12, x3, u, k, 1);\n" \
"}\n" \
"\n" \
"/*__kernel\n" \
"void forward2(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = (2 * mj + i) * VSIZE + l;\n" \
"\n" \
"	uint32_3 u[2]; read2(u, x12, x3, k, m);\n" \
"	frwd2(u, read1(wr12, wr3, sj));\n" \
"	write2(x12, x3, u, k, m);\n" \
"}*/\n" \
"\n" \
"__kernel\n" \
"void forward4(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm, const uint32 reg)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = (reg * NSIZE + (4 * mj + i)) * VSIZE + l;\n" \
"\n" \
"	uint32_3 u[4]; read4(u, x12, x3, k, m);\n" \
"	frwd4(u, wr12, wr3, sj);\n" \
"	write4(x12, x3, u, k, m);\n" \
"}\n" \
"\n" \
"/*__kernel\n" \
"void forward8(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = (8 * mj + i) * VSIZE + l;\n" \
"\n" \
"	uint32_3 u[8]; read8(u, x12, x3, k, m);\n" \
"	frwd8(u, wr12, wr3, sj);\n" \
"	write8(x12, x3, u, k, m);\n" \
"}*/\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16 / 4 * VSIZE, 1, 1)))\n" \
"void forward16(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm, const uint32 reg)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"	const sz_t il = i % VSIZE, iv = i & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	// const sz_t global_id = 16 / 4 * VSIZE * group_id + i, vid = global_id / VSIZE;\n" \
"	const sz_t vid = 16 / 4 * group_id + i / VSIZE;\n" \
"\n" \
"	const sz_t vid_blk = vid & (sz_t)~(4 * m - 1), idl = group_id & (m - 1);\n" \
"	const sz_t k_group = (reg * NSIZE + (4 * vid_blk + idl)) * VSIZE, miv = iv << lm;\n" \
"	const sz_t sj4 = s * 4 + (vid_blk >> lm) + i / VSIZE, sj = sj4 / 4;\n" \
"\n" \
"	uint32_3 u[4]; \n" \
"\n" \
"	read4(u, x12, x3, k_group + miv + il, 4 * m);\n" \
"	frwd4(u, wr12, wr3, sj);\n" \
"	write4l(X12, X3, u, i, 4);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, 4 * iv + il, 1);\n" \
"	frwd4(u, wr12, wr3, sj4);\n" \
"	write4(x12, x3, u, k_group + 4 * miv + il, m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16 / 4 * VSIZE, 1, 1)))\n" \
"void forward16_0(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 reg)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const uint32 m = NSIZE / 16;\n" \
"	const int lm = LNSIZE - 4;\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"	const sz_t il = i % VSIZE, iv = i & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	// const sz_t global_id = 16 / 4 * VSIZE * group_id + i, vid = global_id / VSIZE, l = global_id % VSIZE;\n" \
"	const sz_t vid = 16 / 4 * group_id + i / VSIZE;\n" \
"\n" \
"	const sz_t vid_blk = vid & (sz_t)~(4 * m - 1), idl = group_id & (m - 1);\n" \
"	const sz_t k_group = (reg * NSIZE + (4 * vid_blk + idl)) * VSIZE, miv = iv << lm;\n" \
"	const sz_t sj4 = 4 + (vid_blk >> lm) + i / VSIZE;\n" \
"\n" \
"	uint32_3 u[4];\n" \
"\n" \
"	read4(u, x12, x3, k_group + miv + il, 4 * m);\n" \
"	frwd41_0(u, read1(wr12, wr3, 1));\n" \
"	uint32_3 w_23[2]; w_23[0] = read1(wr12, wr3, 2); w_23[1] = read1(wr12, wr3, 3); \n" \
"	frwd42_0(u, w_23);\n" \
"	write4l(X12, X3, u, iv + il, 4);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, 4 * iv + il, 1);\n" \
"	frwd4(u, wr12, wr3, sj4);\n" \
"	write4(x12, x3, u, k_group + 4 * miv + il, m);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(32 / 4 * VSIZE, 1, 1)))\n" \
"void forward32(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	__local uint32_2 X12[32 * VSIZE];	// VSIZE = 64 => 16 KB\n" \
"	__local uint32 X3[32 * VSIZE];\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"	const sz_t il = i % VSIZE, iv = i & (sz_t)~(VSIZE - 1);\n" \
"	const sz_t il_2 = i % (VSIZE / 2), iv_2 = i & (sz_t)~(VSIZE / 2 - 1);\n" \
"\n" \
"	// const sz_t global_id = 32 / 4 * VSIZE * group_id + i, vid = global_id / VSIZE;\n" \
"	const sz_t vid = 32 / 4 * group_id + i / VSIZE;	// 0 <= vid < 32 / 4\n" \
"\n" \
"	const sz_t vid_blk = vid & (sz_t)~(8 * m - 1), idl = group_id & (m - 1);\n" \
"	const sz_t k_group = (4 * vid_blk + idl) * VSIZE, miv = iv << lm, miv_2 = iv_2 << lm;\n" \
"\n" \
"	const sz_t sj32 = s * 16 + (vid_blk >> (lm - 1)) + i / (VSIZE / 2), sj8 = sj32 / 4, sj2 = sj8 / 4;\n" \
"	const sz_t k8 = 4 * (i & (sz_t)~(2 * VSIZE - 1)) + (i % (2 * VSIZE));\n" \
"\n" \
"	uint32_3 u[4]; \n" \
"\n" \
"	read4(u, x12, x3, k_group + miv + il, 8 * m);\n" \
"	frwd4(u, wr12, wr3, sj2);\n" \
"	write4l(X12, X3, u, i, 8);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, k8, 2);\n" \
"	frwd4(u, wr12, wr3, sj8);\n" \
"	write4l(X12, X3, u, k8, 2);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l_2(u, X12, X3, 4 * iv_2 + il_2);\n" \
"	frwd41(u, read1(wr12, wr3, sj32));\n" \
"	write4_2(x12, x3, u, k_group + 4 * miv_2 + il_2, m);\n" \
"}\n" \
"\n" \
"/*__kernel\n" \
"void backward2(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (2 * mj + i) + l;\n" \
"\n" \
"	uint32_3 u[2]; read2(u, x12, x3, k, m);\n" \
"	bkwd2(u, read1(wri12, wri3, sj));\n" \
"	write2(x12, x3, u, k, m);\n" \
"}*/\n" \
"\n" \
"__kernel\n" \
"void backward4(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (4 * mj + i) + l;\n" \
"\n" \
"	uint32_3 u[4]; read4(u, x12, x3, k, m);\n" \
"	bkwd4(u, wri12, wri3, sj);\n" \
"	write4(x12, x3, u, k, m);\n" \
"}\n" \
"\n" \
"/*__kernel\n" \
"void backward8(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = s + (vid >> lm), i = vid & (m - 1), mj = vid - i, k = VSIZE * (8 * mj + i) + l;\n" \
"\n" \
"	uint32_3 u[8]; read8(u, x12, x3, k, m);\n" \
"	bkwd8(u, wri12, wri3, sj);\n" \
"	write8(x12, x3, u, k, m);\n" \
"}*/\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16 / 4 * VSIZE, 1, 1)))\n" \
"void backward16(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"	const sz_t il = i % VSIZE, iv = i & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	// const sz_t global_id = 16 / 4 * VSIZE * group_id + i, vid = global_id / VSIZE;\n" \
"	const sz_t vid = 16 / 4 * group_id + i / VSIZE;\n" \
"\n" \
"	const sz_t vid_blk = vid & (sz_t)~(4 * m - 1), idl = group_id & (m - 1);\n" \
"	const sz_t k_group = (4 * vid_blk + idl) * VSIZE, miv = iv << lm;\n" \
"	const sz_t sj4 = s * 4 + (vid_blk >> lm) + i / VSIZE, sj = sj4 / 4;\n" \
"\n" \
"	uint32_3 u[4];\n" \
"\n" \
"	read4(u, x12, x3, k_group + 4 * miv + il, m);\n" \
"	bkwd4(u, wri12, wri3, sj4);\n" \
"	write4l(X12, X3, u, 4 * iv + il, 1);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, iv + il, 4);\n" \
"	bkwd4(u, wri12, wri3, sj);\n" \
"	write4(x12, x3, u, k_group + miv + il, 4 * m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((reqd_work_group_size(16 / 4 * VSIZE, 1, 1)))\n" \
"void backward16_0(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const uint32 m = NSIZE / 16;\n" \
"	const int lm = LNSIZE - 4;\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"	const sz_t il = i % VSIZE, iv = i & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	// const sz_t global_id = 16 / 4 * VSIZE * group_id + i, vid = global_id / VSIZE, l = global_id % VSIZE;\n" \
"	const sz_t vid = 16 / 4 * group_id + i / VSIZE;\n" \
"\n" \
"	const sz_t vid_blk = vid & (sz_t)~(4 * m - 1), idl = group_id & (m - 1);\n" \
"	const sz_t k_group = (4 * vid_blk + idl) * VSIZE, miv = iv << lm;\n" \
"	const sz_t sj4 = 4 + (vid_blk >> lm) + i / VSIZE;\n" \
"\n" \
"	uint32_3 u[4];\n" \
"\n" \
"	read4(u, x12, x3, k_group + 4 * miv + il, m);\n" \
"	bkwd4(u, wri12, wri3, sj4);\n" \
"	write4l(X12, X3, u, 4 * iv + il, 1);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, iv + il, 4);\n" \
"	uint32_3 wi_23[2]; wi_23[0] = read1(wri12, wri3, 2); wi_23[1] = read1(wri12, wri3, 3); \n" \
"	bkwd42(u, wi_23);\n" \
"	bkwd41_0(u, read1(wri12, wri3, 0), read1(wri12, wri3, 1));\n" \
"	write4(x12, x3, u, k_group + miv + il, 4 * m);\n" \
"}\n" \
"\n" \
"__kernel // __attribute__((reqd_work_group_size(32 / 4 * VSIZE, 1, 1)))\n" \
"void backward32(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	__local uint32_2 X12[32 * VSIZE];	// VSIZE = 64 => 16 KB\n" \
"	__local uint32 X3[32 * VSIZE];\n" \
"\n" \
"	const sz_t i = (sz_t)get_local_id(0), group_id = (sz_t)get_group_id(0);\n" \
"	const sz_t il = i % VSIZE, iv = i & (sz_t)~(VSIZE - 1);\n" \
"	const sz_t il_2 = i % (VSIZE / 2), iv_2 = i & (sz_t)~(VSIZE / 2 - 1);\n" \
"\n" \
"	// const sz_t global_id = 32 / 4 * VSIZE * group_id + i, vid = global_id / VSIZE;\n" \
"	const sz_t vid = 32 / 4 * group_id + i / VSIZE;	// 0 <= vid < 32 / 4\n" \
"\n" \
"	const sz_t vid_blk = vid & (sz_t)~(8 * m - 1), idl = group_id & (m - 1);\n" \
"	const sz_t k_group = (4 * vid_blk + idl) * VSIZE, miv = iv << lm, miv_2 = iv_2 << lm;\n" \
"\n" \
"	const sz_t sj32 = s * 16 + (vid_blk >> (lm - 1)) + i / (VSIZE / 2), sj8 = sj32 / 4, sj2 = sj8 / 4;\n" \
"	const sz_t k8 = 4 * (i & (sz_t)~(2 * VSIZE - 1)) + (i % (2 * VSIZE));\n" \
"\n" \
"	uint32_3 u[4]; \n" \
"\n" \
"	read4_2(u, x12, x3, k_group + 4 * miv_2 + il_2, m);\n" \
"	bkwd41(u, read1(wri12, wri3, sj32));\n" \
"	write4l_2(X12, X3, u, 4 * iv_2 + il_2);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, k8, 2);\n" \
"	bkwd4(u, wri12, wri3, sj8);\n" \
"	write4l(X12, X3, u, k8, 2);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	read4l(u, X12, X3, i, 8);\n" \
"	bkwd4(u, wri12, wri3, sj2);\n" \
"	write4(x12, x3, u, k_group + miv + il, 8 * m);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize1(const __global uint32_2 * restrict const bb_inv, const __global int32 * restrict const bs,\n" \
"				__global int64 * restrict const f, __global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"				const uint64 dup)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	const sz_t i = id % VSIZE, j = id / VSIZE, k0 = j * (VSIZE * CSIZE) + i;\n" \
"	const uint32 b = bb_inv[i].s0, b_inv = bb_inv[i].s1;\n" \
"	const int32 b_s = bs[i];\n" \
"\n" \
"	int96 a = int96_zero();\n" \
"	for (sz_t c = 0; c < CSIZE; ++c)\n" \
"	{\n" \
"		const sz_t k = k0 + c * VSIZE;\n" \
"		int96 l = garner3(read1(x12, x3, k));\n" \
"		const int96 l2 = ((dup & mask64[i]) != 0) ? l : int96_zero();\n" \
"		a = int96_add(int96_add(a, l), l2);\n" \
"		const int32 r = reduce96(&a, b, b_inv, b_s);\n" \
"		const uint32_3 u = seti_3(r);\n" \
"		x12[k] = (uint32_2)(u.r1, u.r2); x3[k] = u.r3;\n" \
"	}\n" \
"	f[id] = (int64)a.s0;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize2(const __global uint32_2 * restrict const bb_inv, const __global int32 * restrict const bs,\n" \
"				const __global int64 * restrict const f, __global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
"	// get_global_size(0) is VSIZE * NSIZE / CSIZE\n" \
"	const sz_t i = id % VSIZE, j = (id / VSIZE + 1) & (NSIZE / CSIZE - 1);\n" \
"	const uint32 b = bb_inv[i].s0, b_inv = bb_inv[i].s1;\n" \
"	const int32 b_s = bs[i];\n" \
"	int64 a = f[id];\n" \
"	const sz_t k0 = j * (VSIZE * CSIZE) + i;\n" \
"	if (j == 0) a = -a;\n" \
"	else if (j == NSIZE / CSIZE / 2) a += f[id + VSIZE * NSIZE / CSIZE / 2];\n" \
"\n" \
"	sz_t c;\n" \
"	for (c = 0; c < CSIZE - 1; ++c)\n" \
"	{\n" \
"		const sz_t k = k0 + c * VSIZE;\n" \
"		a += geti_P1(x12[k].s0);\n" \
"		const int32 r = reduce64(&a, b, b_inv, b_s);\n" \
"		const uint32_3 u = seti_3(r);\n" \
"		x12[k] = (uint32_2)(u.r1, u.r2); x3[k] = u.r3;\n" \
"		if (a == 0) return;\n" \
"	}\n" \
"\n" \
"	const sz_t k = k0 + c * VSIZE;\n" \
"	const uint32_3 u = add_3(read1(x12, x3, k), seti_3((int32)a));\n" \
"	x12[k] = (uint32_2)(u.r1, u.r2); x3[k] = u.r3;\n" \
"}\n" \
"\n" \
"//////////////////////////////////////////////////////////////////\n" \
"/*\n" \
"__kernel\n" \
"void add_throughput(__global uint32 * restrict const data, __global uint32 * restrict const res)\n" \
"{\n" \
"#ifdef __NV_CL_C_VERSION\n" \
"	uint64 t0 = 0, t1 = 0;\n" \
"	__global uint32 * const pdata = data;\n" \
"	__global uint32 * const pres = res;\n" \
"	uint32 r[16];\n" \
" 	for (sz_t i = 0; i < 16; ++i) r[i] = pdata[i];\n" \
"\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t0) :: \"memory\");\n" \
"	const uint32 p = (uint32)t0;\n" \
"	for (sz_t j = 0; j < 65536/2; ++j)\n" \
"	{\n" \
"		for (sz_t i = 0; i < 8; ++i) r[i] = _addMod(r[i], r[i + 8], p);\n" \
"		for (sz_t i = 0; i < 8; ++i) r[i + 8] = _addMod(r[i + 8], r[i], p);\n" \
"	}\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t1) :: \"memory\");\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"\n" \
"	for (sz_t i = 0; i < 8; ++i) pdata[i] = r[i] + (uint32)t1;\n" \
"	pres[0] = (uint32)(t1 - t0);\n" \
"#else\n" \
"	res[0] = 0;\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void add_latency(__global uint32 * restrict const data, __global uint32 * restrict const res)\n" \
"{\n" \
"#ifdef __NV_CL_C_VERSION\n" \
"	uint64 t0 = 0, t1 = 0;\n" \
"	__global uint32 * const pdata = data;\n" \
"	__global uint32 * const pres = res;\n" \
" 	uint32 r = pdata[0], ra = pdata[1];\n" \
"\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t0) :: \"memory\");\n" \
"	const uint32 p = (uint32)t0;\n" \
"	for (sz_t j = 0; j < 65536/2; ++j)\n" \
"	{\n" \
"		for (sz_t i = 0; i < 8; ++i) { r = _addMod(r, ra, p); ra = _addMod(ra, r, p); }\n" \
"	}\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t1) :: \"memory\");\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"\n" \
"	pdata[0] = r + (uint32)t1;\n" \
"	pres[0] = (uint32)(t1 - t0);\n" \
"#else\n" \
"	res[0] = 0;\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void sub_throughput(__global uint32 * restrict const data, __global uint32 * restrict const res)\n" \
"{\n" \
"#ifdef __NV_CL_C_VERSION\n" \
"	uint64 t0 = 0, t1 = 0;\n" \
"	__global uint32 * const pdata = data;\n" \
"	__global uint32 * const pres = res;\n" \
"	uint32 r[16];\n" \
" 	for (sz_t i = 0; i < 16; ++i) r[i] = pdata[i];\n" \
"\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t0) :: \"memory\");\n" \
"	const uint32 p = (uint32)t0;\n" \
"	for (sz_t j = 0; j < 65536; ++j)\n" \
"	{\n" \
"		for (sz_t i = 0; i < 8; ++i) r[i] = _subMod(r[i], r[i + 8], p);\n" \
"	}\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t1) :: \"memory\");\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"\n" \
"	for (sz_t i = 0; i < 8; ++i) pdata[i] = r[i] + (uint32)t1;\n" \
"	pres[0] = (uint32)(t1 - t0);\n" \
"#else\n" \
"	res[0] = 0;\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void sub_latency(__global uint32 * restrict const data, __global uint32 * restrict const res)\n" \
"{\n" \
"#ifdef __NV_CL_C_VERSION\n" \
"	uint64 t0 = 0, t1 = 0;\n" \
"	__global uint32 * const pdata = data;\n" \
"	__global uint32 * const pres = res;\n" \
" 	uint32 r = pdata[0], rs = pdata[1];\n" \
"\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t0) :: \"memory\");\n" \
"	const uint32 p = (uint32)t0;\n" \
"	for (sz_t j = 0; j < 65536; ++j)\n" \
"	{\n" \
"		for (sz_t i = 0; i < 8; ++i) r = _subMod(r, rs, p);\n" \
"	}\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t1) :: \"memory\");\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"\n" \
"	pdata[0] = r + (uint32)t1;\n" \
"	pres[0] = (uint32)(t1 - t0);\n" \
"#else\n" \
"	res[0] = 0;\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul_throughput(__global uint32 * restrict const data, __global uint32 * restrict const res)\n" \
"{\n" \
"#ifdef __NV_CL_C_VERSION\n" \
"	uint64 t0 = 0, t1 = 0;\n" \
"	__global uint32 * const pdata = data;\n" \
"	__global uint32 * const pres = res;\n" \
"	uint32 r[8];\n" \
" 	for (sz_t i = 0; i < 8; ++i) r[i] = pdata[i];\n" \
"\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t0) :: \"memory\");\n" \
"	const uint32 p = (uint32)t0, q = (uint32)(t0 >> 32);\n" \
"	for (sz_t j = 0; j < 65536; ++j)\n" \
"	{\n" \
"		for (sz_t i = 0; i < 8; ++i) r[i] = _mulMonty(r[i], r[i], p, q);\n" \
"	}\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t1) :: \"memory\");\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"\n" \
"	for (sz_t i = 0; i < 8; ++i) pdata[i] = r[i] + (uint32)t1;\n" \
"	pres[0] = (uint32)(t1 - t0);\n" \
"#else\n" \
"	res[0] = 0;\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul_latency(__global uint32 * restrict const data, __global uint32 * restrict const res)\n" \
"{\n" \
"#ifdef __NV_CL_C_VERSION\n" \
"	uint64 t0 = 0, t1 = 0;\n" \
"	__global uint32 * const pdata = data;\n" \
"	__global uint32 * const pres = res;\n" \
" 	uint32 r = pdata[0];\n" \
"\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t0) :: \"memory\");\n" \
"	const uint32 p = (uint32)t0, q = (uint32)(t0 >> 32);\n" \
"	for (sz_t j = 0; j < 65536; ++j)\n" \
"	{\n" \
"		for (sz_t i = 0; i < 8; ++i) r = _mulMonty(r, r, p, q);\n" \
"	}\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t1) :: \"memory\");\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"\n" \
"	pdata[0] = r + (uint32)t1;\n" \
"	pres[0] = (uint32)(t1 - t0);\n" \
"#else\n" \
"	res[0] = 0;\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void but_throughput(__global uint32 * restrict const data, __global uint32 * restrict const res)\n" \
"{\n" \
"#ifdef __NV_CL_C_VERSION\n" \
"	uint64 t0 = 0, t1 = 0;\n" \
"	__global uint32 * const pdata = data;\n" \
"	__global uint32 * const pres = res;\n" \
"	uint32 r[16];\n" \
" 	for (sz_t i = 0; i < 16; ++i) r[i] = pdata[i];\n" \
"	const uint32 c = pdata[16];\n" \
"\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t0) :: \"memory\");\n" \
"	const uint32 p = (uint32)t0, q = (uint32)(t0 >> 32);\n" \
"	for (sz_t j = 0; j < 65536; ++j)\n" \
"	{\n" \
"		for (sz_t i = 0; i < 8; ++i)\n" \
"		{\n" \
"			const uint32 u0 = r[i + 0], u1 = _mulMonty(r[i + 8], c, p, q);\n" \
"			r[i + 0] = _addMod(u0, u1, p); r[i + 8] = _subMod(u0, u1, p);\n" \
"		}\n" \
"	}\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t1) :: \"memory\");\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"\n" \
"	for (sz_t i = 0; i < 8; ++i) pdata[i] = r[i] + (uint32)t1;\n" \
"	pres[0] = (uint32)(t1 - t0);\n" \
"#else\n" \
"	res[0] = 0;\n" \
"#endif\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void but_latency(__global uint32 * restrict const data, __global uint32 * restrict const res)\n" \
"{\n" \
"#ifdef __NV_CL_C_VERSION\n" \
"	uint64 t0 = 0, t1 = 0;\n" \
"	__global uint32 * const pdata = data;\n" \
"	__global uint32 * const pres = res;\n" \
"	uint32 r[2];\n" \
" 	for (sz_t i = 0; i < 2; ++i) r[i] = pdata[i];\n" \
"	const uint32 c = pdata[2];\n" \
"\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t0) :: \"memory\");\n" \
"	const uint32 p = (uint32)t0, q = (uint32)(t0 >> 32);\n" \
"	for (sz_t j = 0; j < 65536; ++j)\n" \
"	{\n" \
"		for (sz_t i = 0; i < 8; ++i)\n" \
"		{\n" \
"			const uint32 u0 = r[0], u1 = _mulMonty(r[1], c, p, q);\n" \
"			r[0] = _addMod(u0, u1, p); r[1] = _subMod(u0, u1, p);\n" \
"		}\n" \
"	}\n" \
"	asm volatile (\"mov.u64 %0, %%clock64;\" : \"=l\"(t1) :: \"memory\");\n" \
"	barrier(CLK_GLOBAL_MEM_FENCE);\n" \
"\n" \
"	for (sz_t i = 0; i < 2; ++i) pdata[i] = r[i] + (uint32)t1;\n" \
"	pres[0] = (uint32)(t1 - t0);\n" \
"#else\n" \
"	res[0] = 0;\n" \
"#endif\n" \
"}\n" \
"*/\n" \
"";
