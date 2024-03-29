/*
Copyright 2024, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

typedef uint	uint32;
typedef ulong	uint64;

inline int uint64_log2(const uint64 x) { return 63 - clz(x); }

inline uint64 REDC(const uint64 t_lo, const uint64 t_hi, const uint64 p, const uint64 q)
{
	const uint64 mp = mul_hi(t_lo * q, p), r = t_hi - mp;
	return (t_hi < mp) ? r + p : r;
}

inline uint64 REDCshort(const uint64 t, const uint64 p, const uint64 q)
{
	const uint64 mp = mul_hi(t * q, p);
	return (mp != 0) ? p - mp : 0;
}

inline uint64 add_mod(const uint64 a, const uint64 b, const uint64 p)
{
	const uint64 c = (a >= p - b) ? p : 0;
	return a + b - c;
}

inline uint64 sub_mod(const uint64 a, const uint64 b, const uint64 p)
{
	const uint64 c = (a < b) ? p : 0;
	return a - b + c;
}

inline uint64 half_mod(const uint64 a, const uint64 p)
{
	return (a % 2 == 0) ? a / 2 : a / 2 + p / 2 + 1;
}

inline uint64 mul_mod(const uint64 a, const uint64 b, const uint64 p, const uint64 q)
{
	return REDC(a * b, mul_hi(a, b), p, q);
}

// Montgomery form of 2^64 is (2^64)^2
/*inline uint64 two_pow_64(const uint64 p, const uint64 q, const uint64 one)
{
	uint64 t = add_mod(one, one, p); t = add_mod(t, t, p);		// 4
	t = add_mod(t, t, p); t = add_mod(t, t, p);					// 16
	for (size_t i = 0; i < 4; ++i) t = mul_mod(t, t, p, q);		// 16^{2^4} = 2^64
	return t;
}

inline uint64 toMp(const uint64 n, const uint64 p, const uint64 q, const uint64 one)
{
	const uint64 r2 = two_pow_64(p, q, one);
	return mul_mod(n, r2, p, q);
}*/

inline uint64 toInt(const uint64 r, const uint64 p, const uint64 q)
{
	return REDCshort(r, p, q);
}

// p * p_inv = 1 (mod 2^64) (Newton's method)
inline uint64 invert(const uint64 p)
{
	uint64 p_inv = 1, prev = 0;
	while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
	return p_inv;
}

// a^e mod p, left-to-right algorithm
inline uint64 pow_mod(const uint64 a, const uint64 e, const uint64 p, const uint64 q)
{
	uint64 r = a;
	for (int b = uint64_log2(e) - 1; b >= 0; --b)
	{
		r = mul_mod(r, r, p, q);
		if ((e & ((uint64)(1) << b)) != 0) r = mul_mod(r, a, p, q);
	}
	return r;
}

// 2^(p - 1) ?= 1 mod p
inline bool prp(const uint64 p, const uint64 q, const uint64 one)
{
	const uint64 e = (p - 1) / 2;
	int b = uint64_log2(e) - 1;
	uint64 r = add_mod(one, one, p); r = add_mod(r, r, p);
	if ((e & ((uint64)(1) << b)) != 0) r = add_mod(r, r, p);
	for (--b; b >= 0; --b)
	{
		r = mul_mod(r, r, p, q);
		if ((e & ((uint64)(1) << b)) != 0) r = add_mod(r, r, p);
	}
	return ((r == one) || (r == p - one));
}

inline int jacobi(const uint64 x, const uint64 y)
{
	uint64 m = x, n = y;

	int k = 1;
	while (m != 0)
	{
		// (2/n) = (-1)^((n^2-1)/8)
		bool odd = false;
		while (m % 2 == 0) { m /= 2; odd = !odd; }
		if (odd && (n % 8 != 1) && (n % 8 != 7)) k = -k;

		if (m == 1) return k;	// (1/n) = 1

		// (m/n)(n/m) = -1 iif m == n == 3 (mod 4)
		if ((m % 4 == 3) && (n % 4 == 3)) k = -k;
		const uint64 t = n; n = m; m = t;

		m %= n;	// (m/n) = (m mod n / n)
	}

	return 0;	// x and y are not coprime
}

// sqrt for p = 3 (mod 4)
inline uint64 sqrt_mod_3_4(const uint64 a, const uint64 p, const uint64 q)
{
	return pow_mod(a, (p + 1) / 4, p, q);
}

// sqrt for p = 5 (mod 8)
inline uint64 sqrt_mod_5_8(const uint64 a, const uint64 p, const uint64 q, const uint64 one)
{
	const uint64 b = add_mod(a, a, p);
	const uint64 v = pow_mod(b, (p - 5) / 8, p, q);
	const uint64 i = mul_mod(b, mul_mod(v, v, p, q), p, q);	// i^2 = -1
	return mul_mod(mul_mod(a, v, p, q), sub_mod(i, one, p), p, q);
}

// Cipolla's algorithm
inline uint64 sqrt_mod(const uint64 a, const uint64 p, const uint64 q, const uint64 one)
{
	if ((a == 0) || (a == one)) return a;

	uint32 d = 2; uint64 md = add_mod(one, one, p), ai = p - toInt(a, p, q);
	while (d < 256)
	{
		if (jacobi(d * d + ai, p) == -1) break;
		++d; md = add_mod(md, one, p);
	}
	if (d >= 256) return 0;

	const uint64 w2 = sub_mod(mul_mod(md, md, p, q), a, p), e = (p + 1) / 2;
	uint64 x = md, y = one;
	for (int b = uint64_log2(e) - 1; b >= 0; --b)
	{
		const uint64 t = add_mod(mul_mod(x, x, p, q), mul_mod(mul_mod(y, y, p, q), w2, p, q), p); y = mul_mod(add_mod(x, x, p), y, p, q); x = t;
		if ((e & ((uint64)(1) << b)) != 0)
		{
			const uint64 t = add_mod(mul_mod(x, md, p, q), mul_mod(y, w2, p, q), p); y = add_mod(x, mul_mod(y, md, p, q), p); x = t;
		}
	}

	return ((mul_mod(x, x, p, q) == a) ? x : 0);
}

inline uint64 sqrtn_3_4(const uint64 a, const uint32 n, const uint64 p, const uint64 q)
{
	uint64 r = a, k = (p - 1) / 2, m = 1;
	for (uint32 i = 1; i <= n; ++i)
	{
		if (m % 2 != 0) m += k;
		m /= 2;
	}
	if (m > 1) r = pow_mod(r, m, p, q);

	return r;
}

inline uint64 sqrtn_5_8(const uint64 a, const uint32 n, const uint64 p, const uint64 q, const uint64 one)
{
	uint32 e = 1; for (uint64 k = (p - 1) / 2; k % 2 == 0; k /= 2) ++e;
	e = min(e, n);
	if ((e > 1) && pow_mod(a, (p - 1) >> e, p, q) != one) return 0;

	uint64 r = a, k = (p - 1) / 2, m = 1;
	for (uint32 i = 1; i <= n; ++i)
	{
		if (k % 2 == 0)
		{
			r = sqrt_mod_5_8(r, p, q, one);
			k /= 2;
			if (r == 0) break;
		}
		else
		{
			if (m % 2 != 0) m += k;
			m /= 2;
		}
	}
	if (m > 1) r = pow_mod(r, m, p, q);

	return r;
}

inline uint64 sqrtn(const uint64 a, const uint32 n, const uint64 p, const uint64 q, const uint64 one)
{
	uint32 e = 1; for (uint64 k = (p - 1) / 2; k % 2 == 0; k /= 2) ++e;
	e = min(e, n);
	if ((e > 1) && pow_mod(a, (p - 1) >> e, p, q) != one) return 0;

	uint64 r = a, k = (p - 1) / 2, m = 1;
	for (uint32 i = 1; i <= n; ++i)
	{
		if (k % 2 == 0)
		{
			r = sqrt_mod(r, p, q, one);
			k /= 2;
			if (r == 0) break;
		}
		else
		{
			if (m % 2 != 0) m += k;
			m /= 2;
		}
	}
	if (m > 1) r = pow_mod(r, m, p, q);

	return r;
}

inline void push_factor(const uint64 b, uint64 bMax, const uint64 p,
	__global uint * restrict const factor_count, __global ulong2 * restrict const factor)
{
	if (b <= bMax)
	{
		const uint factor_index = atomic_inc(factor_count);
		factor[factor_index] = (ulong2)(p, b);
	}
}

__kernel
void generate_primes_pos(__global uint * restrict const prime_count, __global ulong2 * restrict const prime_vector,
	__global ulong * restrict const ext_vector, const ulong index)
{
	const uint64 k = index | get_global_id(0);

	const uint64 p = 3 * (k << g_n) + 1, q = invert(p), one = (-p) % p;
	if (prp(p, q, one))
	{
		const uint prime_index = atomic_inc(prime_count);
		prime_vector[prime_index] = (ulong2)(p, q);
		ext_vector[prime_index] = one;
	}
}

__kernel
void init_factors_pos(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,
	__global ulong * restrict const ext_vector, __global const char * restrict const kro_vector)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_count) return;

	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;
	const uint64 one = ext_vector[i], two = add_mod(one, one, p), four = add_mod(two, two, p);

	const uint64 pm1_6 = (p - 1) / 6;

	// p = 1 (mod 4). If a is odd then (a/p) = (p/a) = ({p mod a}/a)

	uint32 a = 5; uint64 ma = add_mod(four, one, p);
	while (a < 256)
	{
		if ((kro_vector[(a - 5) * 128 + (p % a)] != 0) && (pow_mod(ma, pm1_6, p, q) != p - one)) break;

		a += 2; ma = add_mod(ma, two, p);

		if ((kro_vector[(a - 5) * 128 + (p % a)] != 0) && (pow_mod(ma, pm1_6, p, q) != p - one)) break;

		a += 4; ma = add_mod(ma, four, p);
	}

	// We have a^{3.k.2^{n-1}} = -1 and a^{k.2^{n-1}} != -1.
	// Then x = (a^k)^{2^{n-1}} = j or 1/j and x^2 - x + 1 = 0.

	ext_vector[i] = (a < 256) ? pow_mod(ma, pm1_6 >> (g_n - 1), p, q) : (ulong)(0);
}

__kernel
void generate_factors_pos(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,
	__global const ulong * restrict const ext_vector, __global uint * restrict const factor_count, __global ulong2 * restrict const factor)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_count) return;

	uint64 b = ext_vector[i];
	if (b == 0) return;

	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;
	const uint64 b2 = mul_mod(b, b, p, q), b4 = mul_mod(b2, b2, p, q);
	b = toInt(b, p, q);

	const uint64 bMax = (p < 10000000000000000ul) ? 5 * 1000000000ul : 10 * 1000000000ul;

	for (uint32 j = 1; j < (3u << (g_n - 1)); j += 6)
	{
		push_factor(b, bMax, p, factor_count, factor);
		push_factor(p - b, bMax, p, factor_count, factor);

		b = mul_mod(b, b4, p, q);		// b = (a^k)^{j + 4}

		push_factor(b, bMax, p, factor_count, factor);
		push_factor(p - b, bMax, p, factor_count, factor);

		b = mul_mod(b, b2, p, q);		// b = (a^k)^{j + 6}
	}

	// ext_vector[i] = b;
}

__kernel
void clear_primes_pos(__global uint * restrict const prime_count)
{
	*prime_count = 0;
}

__kernel
void generate_primes_neg(__global uint * restrict const prime_count, __global ulong2 * restrict const prime_vector, __global ulong2 * restrict const ext_vector,
	__global uint * restrict const prime_3_4_count, __global ulong2 * restrict const prime_3_4_vector, __global ulong2 * restrict const ext_3_4_vector,
	__global uint * restrict const prime_5_8_count, __global ulong2 * restrict const prime_5_8_vector, __global ulong2 * restrict const ext_5_8_vector,
	const ulong index)
{
	const uint64 k = index | get_global_id(0);

	for (uint32 i = 0; i <= 1; ++i)
	{
		const uint64 p = 10 * k + 2 * i - 1, q = invert(p), one = (-p) % p;
		if (prp(p, q, one))
		{
			if (p % 4 == 3)
			{
				const uint prime_index = atomic_inc(prime_3_4_count);
				prime_3_4_vector[prime_index] = (ulong2)(p, q);
				ext_3_4_vector[prime_index] = (ulong2)(one, 0);
			}
			else if (p % 8 == 5)
			{
				const uint prime_index = atomic_inc(prime_5_8_count);
				prime_5_8_vector[prime_index] = (ulong2)(p, q);
				ext_5_8_vector[prime_index] = (ulong2)(one, 0);
			}
			else
			{
				const uint prime_index = atomic_inc(prime_count);
				prime_vector[prime_index] = (ulong2)(p, q);
				ext_vector[prime_index] = (ulong2)(one, 0);
			}
		}
	}
}

__kernel
void init_3_4_factors_neg(__global const uint * restrict const prime_3_4_count, __global const ulong2 * restrict const prime_3_4_vector,
	__global ulong2 * restrict const ext_3_4_vector)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_3_4_count) return;

	const uint64 p = prime_3_4_vector[i].s0, q = prime_3_4_vector[i].s1;
	const uint64 one = ext_3_4_vector[i].s0, two = add_mod(one, one, p), five = add_mod(add_mod(two, two, p), one, p);

	const uint64 s5 = sqrt_mod_3_4(five, p, q);
	if (s5 == 0) { ext_3_4_vector[i] = (ulong2)(0, 0); return; }

	const uint64 r = half_mod(sub_mod(one, s5, p), p);
	const bool is_square = (jacobi(toInt(r, p, q), p) == 1);
	const uint64 rs = is_square ? r : sub_mod(one, r, p);
	const uint64 sn = sqrtn_3_4(rs, g_n - 1, p, q);
	ext_3_4_vector[i] = (ulong2)(sn, 0);
}

__kernel
void init_5_8_factors_neg(__global const uint * restrict const prime_5_8_count, __global const ulong2 * restrict const prime_5_8_vector,
	__global ulong2 * restrict const ext_5_8_vector)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_5_8_count) return;

	const uint64 p = prime_5_8_vector[i].s0, q = prime_5_8_vector[i].s1;
	const uint64 one = ext_5_8_vector[i].s0, two = add_mod(one, one, p), five = add_mod(add_mod(two, two, p), one, p);

	const uint64 s5 = sqrt_mod_5_8(five, p, q, one);
	if (s5 == 0) { ext_5_8_vector[i] = (ulong2)(0, 0); return; }

	const uint64 r = half_mod(sub_mod(one, s5, p), p);
	if (jacobi(toInt(r, p, q), p) != 1) { ext_5_8_vector[i] = (ulong2)(0, 0); return; }
	const uint64 sn1 = sqrtn_5_8(r, g_n - 1, p, q, one);
	const uint64 sn2 = sqrtn_5_8(sub_mod(one, r, p), g_n - 1, p, q, one);
	ext_5_8_vector[i] = (ulong2)(sn1, sn2);
}

__kernel
void init_factors_neg(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,
	__global ulong2 * restrict const ext_vector)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_count) return;

	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;
	const uint64 one = ext_vector[i].s0, two = add_mod(one, one, p), five = add_mod(add_mod(two, two, p), one, p);

	const uint64 s5 = sqrt_mod(five, p, q, one);
	if (s5 == 0) { ext_vector[i] = (ulong2)(0, 0); return; }

	const uint64 r = half_mod(sub_mod(one, s5, p), p);
	if (jacobi(toInt(r, p, q), p) != 1) { ext_vector[i] = (ulong2)(0, 0); return; }
	const uint64 sn1 = sqrtn(r, g_n - 1, p, q, one);
	const uint64 sn2 = sqrtn(sub_mod(one, r, p), g_n - 1, p, q, one);
	ext_vector[i] = (ulong2)(sn1, sn2);
}

__kernel
void generate_factors_neg(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,
	__global const ulong2 * restrict const ext_vector, __global uint * restrict const factor_count, __global ulong2 * restrict const factor)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_count) return;

	uint64 r1 = ext_vector[i].s0, r2 = ext_vector[i].s1;

	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1, one = (-p) % p;

	const uint64 bMax = (p < 10000000000000ul) ? 5 * 1000000000ul : 10 * 1000000000ul;

	// u is a primitive (2^e)th root of unity
	uint32 e = 1; for (uint64 k = (p - 1) / 2; k % 2 == 0; k /= 2) ++e;
	e = min(e, (uint32)(g_n - 1));

	uint32 d = 2; uint64 md = add_mod(one, one, p);
	while (d < 256)
	{
		if (jacobi(d, p) == -1) break;
		++d; md = add_mod(md, one, p); if ((d == 4) || (d == 9)) { ++d; md = add_mod(md, one, p); }
	}
	if (d >= 256) return;

	const uint64 u = pow_mod(md, (p - 1) >> e, p, q);

	if (r1 != 0)
	{
		uint64 b = toInt(r1, p, q);
		for (uint32 i = 0, s = 1u << e; i < s; i += 2)
		{
			push_factor(b, bMax, p, factor_count, factor);
			push_factor(p - b, bMax, p, factor_count, factor);
			b = mul_mod(b, u, p, q);
		}
	}

	if (r2 != 0)
	{
		uint64 b = toInt(r2, p, q);
		for (uint32 i = 0, s = 1u << e; i < s; i += 2)
		{
			push_factor(b, bMax, p, factor_count, factor);
			push_factor(p - b, bMax, p, factor_count, factor);
			b = mul_mod(b, u, p, q);
		}
	}
}

__kernel
void clear_primes_neg(__global uint * restrict const prime_count, __global uint * restrict const _prime_3_4_count, __global uint * restrict const _prime_5_8_count)
{
	*prime_count = 0;
	*_prime_3_4_count = 0;
	*_prime_5_8_count = 0;
}
