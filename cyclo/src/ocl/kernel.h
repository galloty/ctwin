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
"	const uint64 l = (uint32)x * (uint64)y, h = (x >> 32) * y + (l >> 32);\n" \
"	uint96 r; r.s0 = (h << 32) | (uint32)l; r.s1 = (uint32)(h >> 32);\n" \
"	return r;\n" \
"}\n" \
"\n" \
"inline uint96 int96_abs(const int96 x)\n" \
"{\n" \
"	const int96 t = (int96_is_neg(x)) ? int96_neg(x) : x;\n" \
"	return int96_u(t);\n" \
"}\n" \
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
"// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.\n" \
"// Montgomery form (lhs, rhs and output): if 0 <= r < p then f is r * 2^32 mod p\n" \
"inline uint32 _mulMonty(const uint32 lhs, const uint32 rhs, const uint32 p, const uint32 q)\n" \
"{\n" \
"	const uint64 t = lhs * (uint64)rhs;\n" \
"	const uint32 mp = mul_hi((uint32)t * q, p), t_hi = (uint32)(t >> 32), r = t_hi - mp;\n" \
"	return (t_hi < mp) ? r + p : r;\n" \
"}\n" \
"\n" \
"inline uint32 add_P1(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P1); }\n" \
"inline uint32 add_P2(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P2); }\n" \
"inline uint32 add_P3(const uint32 lhs, const uint32 rhs) { return _addMod(lhs, rhs, P3); }\n" \
"\n" \
"inline uint32 sub_P1(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P1); }\n" \
"inline uint32 sub_P2(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P2); }\n" \
"inline uint32 sub_P3(const uint32 lhs, const uint32 rhs) { return _subMod(lhs, rhs, P3); }\n" \
"\n" \
"// Montgomery form\n" \
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
"inline uint32_2 add_P12(const uint32_2 lhs, const uint32_2 rhs) { return (uint32_2)(add_P1(lhs.s0, rhs.s0), add_P2(lhs.s1, rhs.s1)); }\n" \
"inline uint32_2 sub_P12(const uint32_2 lhs, const uint32_2 rhs) { return (uint32_2)(sub_P1(lhs.s0, rhs.s0), sub_P2(lhs.s1, rhs.s1)); }\n" \
"inline uint32_2 mul_P12(const uint32_2 lhs, const uint32_2 rhs) { return (uint32_2)(mul_P1(lhs.s0, rhs.s0), mul_P2(lhs.s1, rhs.s1)); }\n" \
"inline uint32_2 toMonty_P12(const uint32_2 lhs) { return (uint32_2)(toMonty_P1(lhs.s0), toMonty_P2(lhs.s1)); }\n" \
"\n" \
"inline static int96 garner3(const uint32 r1, const uint32 r2, const uint32 r3)\n" \
"{\n" \
"	const uint32 u13 = mul_P1(sub_P1(r1, r3), InvP3_P1);\n" \
"	const uint32 u23 = mul_P2(sub_P2(r2, r3), InvP3_P2);\n" \
"	const uint32 u123 = mul_P1(sub_P1(u13, u23), InvP2_P1);\n" \
"	const uint96 n = uint96_add_64(uint96_mul_64_32(P2P3, u123), u23 * (uint64)P3 + r3);\n" \
"	const uint96 P1P2P3 = uint96_set(P1P2P3l, P1P2P3h), P1P2P3_2 = uint96_set(P1P2P3_2l, P1P2P3_2h);\n" \
"	const int96 r = uint96_is_greater(n, P1P2P3_2) ? uint96_subi(n, P1P2P3) : uint96_i(n);\n" \
"	return r;\n" \
"}\n" \
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
"inline void frwd2_P12(uint32_2 * const u_P12, const uint32_2 w12)\n" \
"{\n" \
"	const uint32_2 u1w_P12 = mul_P12(u_P12[1], w12);\n" \
"	u_P12[1] = sub_P12(u_P12[0], u1w_P12); u_P12[0] = add_P12(u_P12[0], u1w_P12);\n" \
"}\n" \
"inline void frwd2_P3(uint32 * const u_P3, const uint32 w3)\n" \
"{\n" \
"	const uint32 u1w_P3 = mul_P3(u_P3[1], w3);\n" \
"	u_P3[1] = sub_P3(u_P3[0], u1w_P3); u_P3[0] = add_P3(u_P3[0], u1w_P3);\n" \
"}\n" \
"\n" \
"inline void bkwd2_P12(uint32_2 * const u_P12, const uint32_2 wi12)\n" \
"{\n" \
"	const uint32_2 v1_P12 = sub_P12(u_P12[0], u_P12[1]);\n" \
"	u_P12[0] = add_P12(u_P12[0], u_P12[1]); u_P12[1] = mul_P12(v1_P12, wi12);\n" \
"}\n" \
"inline void bkwd2_P3(uint32 * const u_P3, const uint32 wi3)\n" \
"{\n" \
"	const uint32 v1_P3 = sub_P3(u_P3[0], u_P3[1]);\n" \
"	u_P3[0] = add_P3(u_P3[0], u_P3[1]); u_P3[1] = mul_P3(v1_P3, wi3);\n" \
"}\n" \
"\n" \
"inline void sqr2_P12(uint32_2 * const u_P12, const uint32_2 w12)\n" \
"{\n" \
"	const uint32_2 u1w_P12 = mul_P12(u_P12[1], w12);\n" \
"	const uint32_2 v0_P12 = add_P12(mul_P12(u_P12[0], u_P12[0]), mul_P12(u1w_P12, u1w_P12));\n" \
"	const uint32_2 v1_P12 = mul_P12(add_P12(u_P12[0], u_P12[0]), u_P12[1]);\n" \
"	u_P12[0] = add_P12(v0_P12, v0_P12); u_P12[1] = add_P12(v1_P12, v1_P12);\n" \
"}\n" \
"inline void sqr2_P3(uint32 * const u_P3, const uint32 w3)\n" \
"{\n" \
"	const uint32 u1w_P3 = mul_P3(u_P3[1], w3);\n" \
"	const uint32 v0_P3 = add_P3(mul_P3(u_P3[0], u_P3[0]), mul_P3(u1w_P3, u1w_P3));\n" \
"	const uint32 v1_P3 = mul_P3(add_P3(u_P3[0], u_P3[0]), u_P3[1]);\n" \
"	u_P3[0] = add_P3(v0_P3, v0_P3); u_P3[1] = add_P3(v1_P3, v1_P3);\n" \
"}\n" \
"\n" \
"inline void read2_P12(uint32_2 * const u_P12, const __global uint32_2 * const x12, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) u_P12[h] = x12[k + h * VSIZE * m];\n" \
"}\n" \
"inline void read2_P3(uint32 * const u_P3, const __global uint32 * const x3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) u_P3[h] = x3[k + h * VSIZE * m];\n" \
"}\n" \
"\n" \
"inline void write2_P12(__global uint32_2 * const x12, const uint32_2 * const u_P12, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) x12[k + h * VSIZE * m] = u_P12[h];\n" \
"}\n" \
"inline void write2_P3(__global uint32 * const x3, const uint32 * const u_P3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) x3[k + h * VSIZE * m] = u_P3[h];\n" \
"}\n" \
"\n" \
"inline void frwd41_P12(uint32_2 * const u_P12, const uint32_2 w12_1)\n" \
"{\n" \
"	const uint32_2 u2w1_P12 = mul_P12(u_P12[2], w12_1), u3w1_P12 = mul_P12(u_P12[3], w12_1);\n" \
"	u_P12[2] = sub_P12(u_P12[0], u2w1_P12); u_P12[0] = add_P12(u_P12[0], u2w1_P12);\n" \
"	u_P12[3] = sub_P12(u_P12[1], u3w1_P12); u_P12[1] = add_P12(u_P12[1], u3w1_P12);\n" \
"}\n" \
"inline void frwd41_P3(uint32 * const u_P3, const uint32 w3_1)\n" \
"{\n" \
"	const uint32 u2w1_P3 = mul_P3(u_P3[2], w3_1), u3w1_P3 = mul_P3(u_P3[3], w3_1);\n" \
"	u_P3[2] = sub_P3(u_P3[0], u2w1_P3); u_P3[0] = add_P3(u_P3[0], u2w1_P3);\n" \
"	u_P3[3] = sub_P3(u_P3[1], u3w1_P3); u_P3[1] = add_P3(u_P3[1], u3w1_P3);\n" \
"}\n" \
"\n" \
"inline void frwd41_0_P12(uint32_2 * const u_P12, const uint32_2 w12_1)\n" \
"{\n" \
"	const uint32_2 u2w1_P12 = mul_P12(u_P12[2], w12_1), u3w1_P12 = mul_P12(u_P12[3], w12_1);\n" \
"	u_P12[2] = add_P12(u_P12[2], sub_P12(u_P12[0], u2w1_P12)); u_P12[0] = add_P12(u_P12[0], u2w1_P12);\n" \
"	u_P12[3] = add_P12(u_P12[3], sub_P12(u_P12[1], u3w1_P12)); u_P12[1] = add_P12(u_P12[1], u3w1_P12);\n" \
"}\n" \
"inline void frwd41_0_P3(uint32 * const u_P3, const uint32 w3_1)\n" \
"{\n" \
"	const uint32 u2w1_P3 = mul_P3(u_P3[2], w3_1), u3w1_P3 = mul_P3(u_P3[3], w3_1);\n" \
"	u_P3[2] = add_P3(u_P3[2], sub_P3(u_P3[0], u2w1_P3)); u_P3[0] = add_P3(u_P3[0], u2w1_P3);\n" \
"	u_P3[3] = add_P3(u_P3[3], sub_P3(u_P3[1], u3w1_P3)); u_P3[1] = add_P3(u_P3[1], u3w1_P3);\n" \
"}\n" \
"\n" \
"inline void frwd42_P12(uint32_2 * const u_P12, const uint32_2 w12_2, const uint32_2 w12_3)\n" \
"{\n" \
"	const uint32_2 u1w2_P12 = mul_P12(u_P12[1], w12_2), u3w3_P12 = mul_P12(u_P12[3], w12_3);\n" \
"	u_P12[1] = sub_P12(u_P12[0], u1w2_P12); u_P12[0] = add_P12(u_P12[0], u1w2_P12);\n" \
"	u_P12[3] = sub_P12(u_P12[2], u3w3_P12); u_P12[2] = add_P12(u_P12[2], u3w3_P12);\n" \
"}\n" \
"inline void frwd42_P3(uint32 * const u_P3, const uint32 w3_2, const uint32 w3_3)\n" \
"{\n" \
"	const uint32 u1w2_P3 = mul_P3(u_P3[1], w3_2), u3w3_P3 = mul_P3(u_P3[3], w3_3);\n" \
"	u_P3[1] = sub_P3(u_P3[0], u1w2_P3); u_P3[0] = add_P3(u_P3[0], u1w2_P3);\n" \
"	u_P3[3] = sub_P3(u_P3[2], u3w3_P3); u_P3[2] = add_P3(u_P3[2], u3w3_P3);\n" \
"}\n" \
"\n" \
"inline void frwd42_0_P12(uint32_2 * const u_P12, const uint32_2 w12_2, const uint32_2 w12_3)\n" \
"{\n" \
"	const uint32_2 u0_P12 = toMonty_P12(u_P12[0]), u2_P12 = toMonty_P12(u_P12[2]);\n" \
"	const uint32_2 u1w2_P12 = mul_P12(u_P12[1], w12_2), u3w3_P12 = mul_P12(u_P12[3], w12_3);\n" \
"	u_P12[1] = sub_P12(u0_P12, u1w2_P12); u_P12[0] = add_P12(u0_P12, u1w2_P12);\n" \
"	u_P12[3] = sub_P12(u2_P12, u3w3_P12); u_P12[2] = add_P12(u2_P12, u3w3_P12);\n" \
"}\n" \
"inline void frwd42_0_P3(uint32 * const u_P3, const uint32 w3_2, const uint32 w3_3)\n" \
"{\n" \
"	const uint32 u0_P3 = toMonty_P3(u_P3[0]), u2_P3 = toMonty_P3(u_P3[2]);\n" \
"	const uint32 u1w2_P3 = mul_P3(u_P3[1], w3_2), u3w3_P3 = mul_P3(u_P3[3], w3_3);\n" \
"	u_P3[1] = sub_P3(u0_P3, u1w2_P3); u_P3[0] = add_P3(u0_P3, u1w2_P3);\n" \
"	u_P3[3] = sub_P3(u2_P3, u3w3_P3); u_P3[2] = add_P3(u2_P3, u3w3_P3);\n" \
"}\n" \
"\n" \
"inline void bkwd42_P12(uint32_2 * const u_P12, const uint32_2 wi12_2, const uint32_2 wi12_3)\n" \
"{\n" \
"	const uint32_2 v1_P12 = sub_P12(u_P12[0], u_P12[1]), v3_P12 = sub_P12(u_P12[2], u_P12[3]);\n" \
"	u_P12[0] = add_P12(u_P12[0], u_P12[1]); u_P12[1] = mul_P12(v1_P12, wi12_2);\n" \
"	u_P12[2] = add_P12(u_P12[2], u_P12[3]); u_P12[3] = mul_P12(v3_P12, wi12_3);\n" \
"}\n" \
"inline void bkwd42_P3(uint32 * const u_P3, const uint32 wi3_2, const uint32 wi3_3)\n" \
"{\n" \
"	const uint32 v1_P3 = sub_P3(u_P3[0], u_P3[1]), v3_P3 = sub_P3(u_P3[2], u_P3[3]);\n" \
"	u_P3[0] = add_P3(u_P3[0], u_P3[1]); u_P3[1] = mul_P3(v1_P3, wi3_2);\n" \
"	u_P3[2] = add_P3(u_P3[2], u_P3[3]); u_P3[3] = mul_P3(v3_P3, wi3_3);\n" \
"}\n" \
"\n" \
"inline void bkwd41_P12(uint32_2 * const u_P12, const uint32_2 wi12_1)\n" \
"{\n" \
"	const uint32_2 v2_P12 = sub_P12(u_P12[0], u_P12[2]), v3_P12 = sub_P12(u_P12[1], u_P12[3]);\n" \
"	u_P12[0] = add_P12(u_P12[0], u_P12[2]); u_P12[2] = mul_P12(v2_P12, wi12_1);\n" \
"	u_P12[1] = add_P12(u_P12[1], u_P12[3]); u_P12[3] = mul_P12(v3_P12, wi12_1);\n" \
"}\n" \
"inline void bkwd41_P3(uint32 * const u_P3, const uint32 wi3_1)\n" \
"{\n" \
"	const uint32 v2_P3 = sub_P3(u_P3[0], u_P3[2]), v3_P3 = sub_P3(u_P3[1], u_P3[3]);\n" \
"	u_P3[0] = add_P3(u_P3[0], u_P3[2]); u_P3[2] = mul_P3(v2_P3, wi3_1);\n" \
"	u_P3[1] = add_P3(u_P3[1], u_P3[3]); u_P3[3] = mul_P3(v3_P3, wi3_1);\n" \
"}\n" \
"\n" \
"inline void bkwd41_0_P12(uint32_2 * const u_P12, const uint32_2 wi12_0, const uint32_2 wi12_1)\n" \
"{\n" \
"	const uint32_2 v2_P12 = mul_P12(add_P12(u_P12[0], u_P12[2]), wi12_0);\n" \
"	const uint32_2 v3_P12 = mul_P12(add_P12(u_P12[1], u_P12[3]), wi12_0);\n" \
"	u_P12[2] = mul_P12(sub_P12(u_P12[2], u_P12[0]), wi12_1);\n" \
"	u_P12[0] = mul_P12(sub_P12(u_P12[0], v2_P12), wi12_1);\n" \
"	u_P12[3] = mul_P12(sub_P12(u_P12[3], u_P12[1]), wi12_1);\n" \
"	u_P12[1] = mul_P12(sub_P12(u_P12[1], v3_P12), wi12_1);\n" \
"}\n" \
"inline void bkwd41_0_P3(uint32 * const u_P3, const uint32 wi3_0, const uint32 wi3_1)\n" \
"{\n" \
"	const uint32 v2_P3 = mul_P3(add_P3(u_P3[0], u_P3[2]), wi3_0);\n" \
"	const uint32 v3_P3 = mul_P3(add_P3(u_P3[1], u_P3[3]), wi3_0);\n" \
"	u_P3[2] = mul_P3(sub_P3(u_P3[2], u_P3[0]), wi3_1);\n" \
"	u_P3[0] = mul_P3(sub_P3(u_P3[0], v2_P3), wi3_1);\n" \
"	u_P3[3] = mul_P3(sub_P3(u_P3[3], u_P3[1]), wi3_1);\n" \
"	u_P3[1] = mul_P3(sub_P3(u_P3[1], v3_P3), wi3_1);\n" \
"}\n" \
"\n" \
"inline void sqr42_P12(uint32_2 * const u_P12, const uint32_2 w12_2, const uint32_2 w12_3)\n" \
"{\n" \
"	const uint32_2 u1w2_P12 = mul_P12(u_P12[1], w12_2), u3w3_P12 = mul_P12(u_P12[3], w12_3);\n" \
"	const uint32_2 v0_P12 = add_P12(mul_P12(u_P12[0], u_P12[0]), mul_P12(u1w2_P12, u1w2_P12));\n" \
"	const uint32_2 v1_P12 = mul_P12(add_P12(u_P12[0], u_P12[0]), u_P12[1]);\n" \
"	const uint32_2 v2_P12 = add_P12(mul_P12(u_P12[2], u_P12[2]), mul_P12(u3w3_P12, u3w3_P12));\n" \
"	const uint32_2 v3_P12 = mul_P12(add_P12(u_P12[2], u_P12[2]), u_P12[3]);\n" \
"	u_P12[0] = add_P12(v0_P12, v0_P12); u_P12[1] = add_P12(v1_P12, v1_P12);\n" \
"	u_P12[2] = add_P12(v2_P12, v2_P12); u_P12[3] = add_P12(v3_P12, v3_P12);\n" \
"}\n" \
"inline void sqr42_P3(uint32 * const u_P3, const uint32 w3_2, const uint32 w3_3)\n" \
"{\n" \
"	const uint32 u1w2_P3 = mul_P3(u_P3[1], w3_2), u3w3_P3 = mul_P3(u_P3[3], w3_3);\n" \
"	const uint32 v0_P3 = add_P3(mul_P3(u_P3[0], u_P3[0]), mul_P3(u1w2_P3, u1w2_P3));\n" \
"	const uint32 v1_P3 = mul_P3(add_P3(u_P3[0], u_P3[0]), u_P3[1]);\n" \
"	const uint32 v2_P3 = add_P3(mul_P3(u_P3[2], u_P3[2]), mul_P3(u3w3_P3, u3w3_P3));\n" \
"	const uint32 v3_P3 = mul_P3(add_P3(u_P3[2], u_P3[2]), u_P3[3]);\n" \
"	u_P3[0] = add_P3(v0_P3, v0_P3); u_P3[1] = add_P3(v1_P3, v1_P3);\n" \
"	u_P3[2] = add_P3(v2_P3, v2_P3); u_P3[3] = add_P3(v3_P3, v3_P3);\n" \
"}\n" \
"\n" \
"inline void read4_P12(uint32_2 * const u_P12, const __global uint32_2 * const x12, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) u_P12[h] = x12[k + h * VSIZE * m];\n" \
"}\n" \
"inline void read4_P3(uint32 * const u_P3, const __global uint32 * const x3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) u_P3[h] = x3[k + h * VSIZE * m];\n" \
"}\n" \
"\n" \
"inline void write4_P12(__global uint32_2 * const x12, const uint32_2 * const u_P12, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) x12[k + h * VSIZE * m] = u_P12[h];\n" \
"}\n" \
"inline void write4_P3(__global uint32 * const x3, const uint32 * const u_P3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) x3[k + h * VSIZE * m] = u_P3[h];\n" \
"}\n" \
"\n" \
"inline void read22l_P12(uint32_2 * const u_P12, const __local uint32_2 * const X12, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) u_P12[h] = X12[k + h * VSIZE * m];\n" \
"	for (sz_t h = 0; h < 2; ++h) u_P12[h + 2] = X12[k + 1 + h * VSIZE * m];\n" \
"}\n" \
"inline void read22l_P3(uint32 * const u_P3, const __local uint32 * const X3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) u_P3[h] = X3[k + h * VSIZE * m];\n" \
"	for (sz_t h = 0; h < 2; ++h) u_P3[h + 2] = X3[k + 1 + h * VSIZE * m];\n" \
"}\n" \
"\n" \
"inline void write22l_P12(__local uint32_2 * const X12, const uint32_2 * const u_P12, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) X12[k + h * VSIZE * m] = u_P12[h];\n" \
"	for (sz_t h = 0; h < 2; ++h) X12[k + 1 + h * VSIZE * m] = u_P12[h + 2];\n" \
"}\n" \
"inline void write22l_P3(__local uint32 * const X3, const uint32 * const u_P3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 2; ++h) X3[k + h * VSIZE * m] = u_P3[h];\n" \
"	for (sz_t h = 0; h < 2; ++h) X3[k + 1 + h * VSIZE * m] = u_P3[h + 2];\n" \
"}\n" \
"\n" \
"inline void read4l_P12(uint32_2 * const u_P12, const __local uint32_2 * const X12, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) u_P12[h] = X12[k + h * VSIZE * m];\n" \
"}\n" \
"inline void read4l_P3(uint32 * const u_P3, const __local uint32 * const X3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) u_P3[h] = X3[k + h * VSIZE * m];\n" \
"}\n" \
"\n" \
"inline void write4l_P12(__local uint32_2 * const X12, const uint32_2 * const u_P12, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) X12[k + h * VSIZE * m] = u_P12[h];\n" \
"}\n" \
"inline void write4l_P3(__local uint32 * const X3, const uint32 * const u_P3, const sz_t k, const uint32 m)\n" \
"{\n" \
"	for (sz_t h = 0; h < 4; ++h) X3[k + h * VSIZE * m] = u_P3[h];\n" \
"}\n" \
"\n" \
"inline void frwd4_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const x12,\n" \
"	__local uint32_2 * const X12, const sz_t kg, const sz_t k, const sz_t m, const sz_t sj)\n" \
"{\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, kg + k, m);\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"	write4l_P12(X12, u_P12, k, m);\n" \
"}\n" \
"inline void frwd4_P3(const __global uint32 * restrict const wr3, const __global uint32 * restrict const x3,\n" \
"	__local uint32 * const X3, const sz_t kg, const sz_t k, const sz_t m, const sz_t sj)\n" \
"{\n" \
"	uint32 u_P3[4]; read4_P3(u_P3, x3, kg + k, m);\n" \
"	frwd41_P3(u_P3, wr3[sj]);\n" \
"	frwd42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"	write4l_P3(X3, u_P3, k, m);\n" \
"}\n" \
"\n" \
"inline void bkwd4_P12(const __global uint32_2 * restrict const wri12, const __local uint32_2 * const X12,\n" \
"	__global uint32_2 * restrict const x12, const sz_t kg, const sz_t k, const sz_t m, const sz_t sj)\n" \
"{\n" \
"	uint32_2 u_P12[4]; read4l_P12(u_P12, X12, k, m);\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"	write4_P12(x12, u_P12, kg + k, m);\n" \
"}\n" \
"inline void bkwd4_P3(const __global uint32 * restrict const wri3, const __local uint32 * const X3,\n" \
"	__global uint32 * restrict const x3, const sz_t kg, const sz_t k, const sz_t m, const sz_t sj)\n" \
"{\n" \
"	uint32 u_P3[4]; read4l_P3(u_P3, X3, k, m);\n" \
"	bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P3(u_P3, wri3[sj]);\n" \
"	write4_P3(x3, u_P3, kg + k, m);\n" \
"}\n" \
"\n" \
"inline void sqr4l_P12(const __global uint32_2 * restrict const wr12, const __global uint32_2 * restrict const wri12,\n" \
"	__local uint32_2 * const X12, const sz_t k, const sz_t sj)\n" \
"{\n" \
"	uint32_2 u_P12[4]; read4l_P12(u_P12, X12, k, 1);\n" \
"	frwd41_P12(u_P12, wr12[sj]);\n" \
"	sqr42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]);\n" \
"	write4l_P12(X12, u_P12, k, 1);\n" \
"}\n" \
"inline void sqr4l_P3(const __global uint32 * restrict const wr3, const __global uint32 * restrict const wri3,\n" \
"	__local uint32 * const X3, const sz_t k, const sz_t sj)\n" \
"{\n" \
"	uint32 u_P3[4]; read4l_P3(u_P3, X3, k, 1);\n" \
"	frwd41_P3(u_P3, wr3[sj]);\n" \
"	sqr42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"	bkwd41_P3(u_P3, wri3[sj]);\n" \
"	write4l_P3(X3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"inline void sqr22l_P12(const __global uint32_2 * restrict const wr12,\n" \
"	__local uint32_2 * const X12, const sz_t k, const sz_t sj)\n" \
"{\n" \
"	uint32_2 u_P12[4]; read22l_P12(u_P12, X12, k, 1);\n" \
"	const uint32_2 w12 = wr12[sj];\n" \
"	sqr2_P12(&u_P12[0], w12); sqr2_P12(&u_P12[2], w12);\n" \
"	write22l_P12(X12, u_P12, k, 1);\n" \
"}\n" \
"inline void sqr22l_P3(const __global uint32 * restrict const wr3,\n" \
"	__local uint32 * const X3, const sz_t k, const sz_t sj)\n" \
"{\n" \
"	uint32 u_P3[4]; read22l_P3(u_P3, X3, k, 1);\n" \
"	const uint32 w3 = wr3[sj];\n" \
"	sqr2_P3(&u_P3[0], w3); sqr2_P3(&u_P3[2], w3);\n" \
"	write22l_P3(X3, u_P3, k, 1);\n" \
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
"	const sz_t sj = (sz_t)get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1); uint32 u_P3[2]; read2_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	sqr2_P12(u_P12, wr12[sj]); sqr2_P3(u_P3, wr3[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1); write2_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void square4(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = (sz_t)get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1); uint32 u_P3[4]; read4_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]); frwd41_P3(u_P3, wr3[sj]);\n" \
"	sqr42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]); sqr42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1); write4_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(8 / 4 * VSIZE, 1, 1)))\n" \
"void square8(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32_2 X12[8 * VSIZE];\n" \
"	__local uint32 X3[8 * VSIZE];\n" \
"\n" \
"	const sz_t n_2 = (sz_t)get_global_size(0) * 2 / VSIZE, sj2 = n_2 + (sz_t)get_global_id(0) * 2 / VSIZE;\n" \
"	const sz_t k_group = (sz_t)get_group_id(0) * 8 * VSIZE, i = (sz_t)get_local_id(0);\n" \
"\n" \
"	const sz_t sj8 = sj2 / 4, k8 = i;\n" \
"	const sz_t k2 = 2 * ((2 * i) & (sz_t)~(VSIZE - 1)) + ((2 * i) % VSIZE);\n" \
"\n" \
"	frwd4_P12(wr12, x12, X12, k_group, k8, 2, sj8); frwd4_P3(wr3, x3, X3, k_group, k8, 2, sj8);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	sqr22l_P12(wr12, X12, k2, sj2); sqr22l_P3(wr3, X3, k2, sj2);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	bkwd4_P12(wri12, X12, x12, k_group, k8, 2, sj8); bkwd4_P3(wri3, X3, x3, k_group, k8, 2, sj8);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void square16(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const sz_t n_4 = (sz_t)get_global_size(0) / VSIZE, sj4 = n_4 + (sz_t)get_global_id(0) / VSIZE;\n" \
"	const sz_t k_group = (sz_t)get_group_id(0) * 16 * VSIZE, i = (sz_t)get_local_id(0);\n" \
"\n" \
"	const sz_t sj16 = sj4 / 4, k16 = i;\n" \
"	const sz_t k4 = 4 * (i & (sz_t)~(VSIZE - 1)) + (i % VSIZE);\n" \
"\n" \
"	frwd4_P12(wr12, x12, X12, k_group, k16, 4, sj16); frwd4_P3(wr3, x3, X3, k_group, k16, 4, sj16);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	sqr4l_P12(wr12, wri12, X12, k4, sj4); sqr4l_P3(wr3, wri3, X3, k4, sj4);\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"	bkwd4_P12(wri12, X12, x12, k_group, k16, 4, sj16); bkwd4_P3(wri3, X3, x3, k_group, k16, 4, sj16);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul2(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12,  __global uint32 * restrict const x3)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = (sz_t)get_global_size(0) / VSIZE + vid, k = VSIZE * 2 * vid + l;\n" \
"\n" \
"	const uint32_2 wr12_1 = wr12[sj];\n" \
"	const uint32 wr3_1 = wr3[sj];\n" \
"\n" \
"	uint32_2 u_P12[2]; read2_P12(u_P12, x12, k, 1); uint32 u_P3[2]; read2_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd2_P12(u_P12, wr12_1); frwd2_P3(u_P3, wr3_1);\n" \
"\n" \
"	uint32_2 v_P12[2]; read2_P12(v_P12, x12, VSIZE * NSIZE + k, 1); uint32 v_P3[2]; read2_P3(v_P3, x3, VSIZE * NSIZE + k, 1);\n" \
"\n" \
"	frwd2_P12(v_P12, wr12_1); frwd2_P3(v_P3, wr3_1);\n" \
"\n" \
"	for (sz_t h = 0; h < 2; ++h) u_P12[h] = mul_P12(u_P12[h], v_P12[h]);\n" \
"	for (sz_t h = 0; h < 2; ++h) u_P3[h] = mul_P3(u_P3[h], v_P3[h]);\n" \
"\n" \
"	bkwd2_P12(u_P12, wri12[sj]); bkwd2_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write2_P12(x12, u_P12, k, 1); write2_P3(x3, u_P3, k, 1);\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void mul4(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	const sz_t id = (sz_t)get_global_id(0), vid = id / VSIZE, l = id % VSIZE;\n" \
"\n" \
"	const sz_t sj = (sz_t)get_global_size(0) / VSIZE + vid, k = VSIZE * 4 * vid + l;\n" \
"\n" \
"	const uint32_2 wr12_1 = wr12[sj], wr12_2 = wr12[2 * sj], wr12_3 = wr12[2 * sj + 1];\n" \
"	const uint32 wr3_1 = wr3[sj], wr3_2 = wr3[2 * sj], wr3_3 = wr3[2 * sj + 1];\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, 1); uint32 u_P3[4]; read4_P3(u_P3, x3, k, 1);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12_1); frwd41_P3(u_P3, wr3_1);\n" \
"	frwd42_P12(u_P12, wr12_2, wr12_3); frwd42_P3(u_P3, wr3_2, wr3_3);\n" \
"\n" \
"	uint32_2 v_P12[4]; read4_P12(v_P12, x12, VSIZE * NSIZE + k, 1); uint32 v_P3[4]; read4_P3(v_P3, x3, VSIZE * NSIZE + k, 1);\n" \
"\n" \
"	frwd41_P12(v_P12, wr12_1); frwd41_P3(v_P3, wr3_1);\n" \
"	frwd42_P12(v_P12, wr12_2, wr12_3); frwd42_P3(v_P3, wr3_2, wr3_3);\n" \
"\n" \
"	for (sz_t h = 0; h < 4; ++h) u_P12[h] = mul_P12(u_P12[h], v_P12[h]);\n" \
"	for (sz_t h = 0; h < 4; ++h) u_P3[h] = mul_P3(u_P3[h], v_P3[h]);\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]); bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, 1); write4_P3(x3, u_P3, k, 1);\n" \
"}\n" \
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
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, m); uint32 u_P3[4]; read4_P3(u_P3, x3, k, m);\n" \
"\n" \
"	frwd41_P12(u_P12, wr12[sj]); frwd41_P3(u_P3, wr3[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]); frwd42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, m); write4_P3(x3, u_P3, k, m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void forward16(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm, const uint32 reg)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), vid = gid / VSIZE, l = gid % VSIZE;\n" \
"	const sz_t lid = (sz_t)get_local_id(0), i = lid / VSIZE, iv = lid & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	const sz_t vid_blk = (vid & (sz_t)~(4 * m - 1)) * 4, idl = get_group_id(0) & (m - 1);\n" \
"	const sz_t k0 = (reg * NSIZE + (vid_blk + idl)) * VSIZE + l, miv = iv << lm;\n" \
"	const sz_t sj4 = s * 4 + (vid_blk >> (lm + 2)) + i, sj = sj4 / 4;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k0 + miv, 4 * m); uint32 u_P3[4]; read4_P3(u_P3, x3, k0 + miv, 4 * m);\n" \
"	frwd41_P12(u_P12, wr12[sj]); frwd41_P3(u_P3, wr3[sj]);\n" \
"	frwd42_P12(u_P12, wr12[2 * sj], wr12[2 * sj + 1]); frwd42_P3(u_P3, wr3[2 * sj], wr3[2 * sj + 1]);\n" \
"	write4l_P12(X12, u_P12, iv + l, 4); write4l_P3(X3, u_P3, iv + l, 4);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	uint32_2 v_P12[4]; read4l_P12(v_P12, X12, 4 * iv + l, 1); uint32 v_P3[4]; read4l_P3(v_P3, X3, 4 * iv + l, 1);\n" \
"	frwd41_P12(v_P12, wr12[sj4]); frwd41_P3(v_P3, wr3[sj4]); frwd42_P3(v_P3, wr3[2 * sj4], wr3[2 * sj4 + 1]);\n" \
"	frwd42_P12(v_P12, wr12[2 * sj4], wr12[2 * sj4 + 1]); write4_P3(x3, v_P3, k0 + 4 * miv, m);\n" \
"	write4_P12(x12, v_P12, k0 + 4 * miv, m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void forward16_0(const __global uint32_2 * restrict const wr12, const __global uint32 * restrict const wr3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3, const uint32 reg)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const uint32 m = NSIZE / 16;\n" \
"	const int lm = LNSIZE - 4;\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), vid = gid / VSIZE, l = gid % VSIZE;\n" \
"	const sz_t lid = (sz_t)get_local_id(0), i = lid / VSIZE, iv = lid & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	const sz_t vid_blk = (vid & (sz_t)~(4 * m - 1)) * 4, idl = get_group_id(0) & (m - 1);\n" \
"	const sz_t k0 = (reg * NSIZE + (vid_blk + idl)) * VSIZE + l, miv = iv << lm;\n" \
"	const sz_t sj4 = 4 + (vid_blk >> (lm + 2)) + i;\n" \
"\n" \
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k0 + miv, 4 * m); uint32 u_P3[4]; read4_P3(u_P3, x3, k0 + miv, 4 * m);\n" \
"	frwd41_0_P12(u_P12, wr12[1]); frwd41_0_P3(u_P3, wr3[1]);\n" \
"	frwd42_0_P12(u_P12, wr12[2], wr12[3]); frwd42_0_P3(u_P3, wr3[2], wr3[3]);\n" \
"	write4l_P12(X12, u_P12, iv + l, 4); write4l_P3(X3, u_P3, iv + l, 4);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	uint32_2 v_P12[4]; read4l_P12(v_P12, X12, 4 * iv + l, 1); uint32 v_P3[4]; read4l_P3(v_P3, X3, 4 * iv + l, 1);\n" \
"	frwd41_P12(v_P12, wr12[sj4]); frwd41_P3(v_P3, wr3[sj4]); frwd42_P3(v_P3, wr3[2 * sj4], wr3[2 * sj4 + 1]);\n" \
"	frwd42_P12(v_P12, wr12[2 * sj4], wr12[2 * sj4 + 1]); write4_P3(x3, v_P3, k0 + 4 * miv, m);\n" \
"	write4_P12(x12, v_P12, k0 + 4 * miv, m);\n" \
"}\n" \
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
"	uint32_2 u_P12[4]; read4_P12(u_P12, x12, k, m); uint32 u_P3[4]; read4_P3(u_P3, x3, k, m);\n" \
"\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]); bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"\n" \
"	write4_P12(x12, u_P12, k, m); write4_P3(x3, u_P3, k, m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void backward16(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3,\n" \
"	const uint32 s, const uint32 m, const int lm)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), vid = gid / VSIZE, l = gid % VSIZE;\n" \
"	const sz_t lid = (sz_t)get_local_id(0), i = lid / VSIZE, iv = lid & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	const sz_t vid_blk = (vid & (sz_t)~(4 * m - 1)) * 4, idl = get_group_id(0) & (m - 1);\n" \
"	const sz_t k0 = VSIZE * (vid_blk + idl) + l, miv = iv << lm;\n" \
"	const sz_t sj4 = s * 4 + (vid_blk >> (lm + 2)) + i, sj = sj4 / 4;\n" \
"\n" \
"	uint32_2 v_P12[4]; read4_P12(v_P12, x12, k0 + 4 * miv, m); uint32 v_P3[4]; read4_P3(v_P3, x3, k0 + 4 * miv, m);\n" \
"	bkwd42_P12(v_P12, wri12[2 * sj4], wri12[2 * sj4 + 1]); bkwd42_P3(v_P3, wri3[2 * sj4], wri3[2 * sj4 + 1]);\n" \
"	bkwd41_P12(v_P12, wri12[sj4]); bkwd41_P3(v_P3, wri3[sj4]);\n" \
"	write4l_P12(X12, v_P12, 4 * iv + l, 1); write4l_P3(X3, v_P3, 4 * iv + l, 1);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	uint32_2 u_P12[4]; read4l_P12(u_P12, X12, iv + l, 4); uint32 u_P3[4]; read4l_P3(u_P3, X3, iv + l, 4);\n" \
"	bkwd42_P12(u_P12, wri12[2 * sj], wri12[2 * sj + 1]); bkwd42_P3(u_P3, wri3[2 * sj], wri3[2 * sj + 1]);\n" \
"	bkwd41_P12(u_P12, wri12[sj]); bkwd41_P3(u_P3, wri3[sj]);\n" \
"	write4_P12(x12, u_P12, k0 + miv, 4 * m); write4_P3(x3, u_P3, k0 + miv, 4 * m);\n" \
"}\n" \
"\n" \
"__kernel __attribute__((work_group_size_hint(16 / 4 * VSIZE, 1, 1)))\n" \
"void backward16_0(const __global uint32_2 * restrict const wri12, const __global uint32 * restrict const wri3,\n" \
"	__global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	__local uint32_2 X12[16 * VSIZE];	// VSIZE = 64 => 8 KB\n" \
"	__local uint32 X3[16 * VSIZE];\n" \
"\n" \
"	const uint32 m = NSIZE / 16;\n" \
"	const int lm = LNSIZE - 4;\n" \
"\n" \
"	const sz_t gid = (sz_t)get_global_id(0), vid = gid / VSIZE, l = gid % VSIZE;\n" \
"	const sz_t lid = (sz_t)get_local_id(0), i = lid / VSIZE, iv = lid & (sz_t)~(VSIZE - 1);\n" \
"\n" \
"	const sz_t vid_blk = (vid & (sz_t)~(4 * m - 1)) * 4, idl = get_group_id(0) & (m - 1);\n" \
"	const sz_t k0 = VSIZE * (vid_blk + idl) + l, miv = iv << lm;\n" \
"	const sz_t sj4 = 4 + (vid_blk >> (lm + 2)) + i;\n" \
"\n" \
"	uint32_2 v_P12[4]; read4_P12(v_P12, x12, k0 + 4 * miv, m); uint32 v_P3[4]; read4_P3(v_P3, x3, k0 + 4 * miv, m);\n" \
"	bkwd42_P12(v_P12, wri12[2 * sj4], wri12[2 * sj4 + 1]); bkwd42_P3(v_P3, wri3[2 * sj4], wri3[2 * sj4 + 1]);\n" \
"	bkwd41_P12(v_P12, wri12[sj4]); bkwd41_P3(v_P3, wri3[sj4]);\n" \
"	write4l_P12(X12, v_P12, 4 * iv + l, 1); write4l_P3(X3, v_P3, 4 * iv + l, 1);\n" \
"\n" \
"	barrier(CLK_LOCAL_MEM_FENCE);\n" \
"\n" \
"	uint32_2 u_P12[4]; read4l_P12(u_P12, X12, iv + l, 4); uint32 u_P3[4]; read4l_P3(u_P3, X3, iv + l, 4);\n" \
"	bkwd42_P12(u_P12, wri12[2], wri12[3]); bkwd42_P3(u_P3, wri3[2], wri3[3]);\n" \
"	bkwd41_0_P12(u_P12, wri12[0], wri12[1]); bkwd41_0_P3(u_P3, wri3[0], wri3[1]);\n" \
"	write4_P12(x12, u_P12, k0 + miv, 4 * m); write4_P3(x3, u_P3, k0 + miv, 4 * m);\n" \
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
"		const uint32_2 x12k = x12[k];\n" \
"		int96 l = garner3(mul_P1(x12k.s0, NORM1), mul_P2(x12k.s1, NORM2), mul_P3(x3[k], NORM3));\n" \
"		const int96 l2 = ((dup & mask64[i]) != 0) ? l : int96_zero();\n" \
"		a = int96_add(int96_add(a, l), l2);\n" \
"		const int32 r = reduce96(&a, b, b_inv, b_s);\n" \
"		x12[k] = (uint32_2)(seti_P1(r), seti_P2(r)); x3[k] = seti_P3(r);\n" \
"	}\n" \
"	f[id] = (int64)a.s0;\n" \
"}\n" \
"\n" \
"__kernel\n" \
"void normalize2(const __global uint32_2 * restrict const bb_inv, const __global int32 * restrict const bs,\n" \
"				const __global int64 * restrict const f, __global uint32_2 * restrict const x12, __global uint32 * restrict const x3)\n" \
"{\n" \
"	// get_global_size(0) is VSIZE * NSIZE / CSIZE\n" \
"	const sz_t id = (sz_t)get_global_id(0);\n" \
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
"		x12[k] = (uint32_2)(seti_P1(r), seti_P2(r)); x3[k] = seti_P3(r);\n" \
"		if (a == 0) return;\n" \
"	}\n" \
"\n" \
"	const sz_t k = k0 + c * VSIZE;\n" \
"	x12[k] = add_P12(x12[k], (uint32_2)(seti_P1((int32)a), seti_P2((int32)a))); x3[k] = add_P3(x3[k], seti_P3((int32)a));\n" \
"}\n" \
"";
