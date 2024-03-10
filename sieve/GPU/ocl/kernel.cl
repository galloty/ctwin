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
}*/

/*inline uint64 toMp(const uint64 n, const uint64 p, const uint64 q, const uint64 one)
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

__kernel
void generate_primes(__global uint * restrict const prime_count, __global ulong2 * restrict const prime_vector,
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
void init_factors(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,
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
void generate_factors(__global const uint * restrict const prime_count, __global const ulong2 * restrict const prime_vector,
	__global const ulong * restrict const ext_vector, __global uint * restrict const factor_count, __global ulong2 * restrict const factor)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_count) return;

	uint64 b = ext_vector[i];
	if (b == 0) return;

	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;
	const uint64 b2 = mul_mod(b, b, p, q), b4 = mul_mod(b2, b2, p, q);
	b = toInt(b, p, q);

	for (uint32 j = 1; j < (3u << (g_n - 1)); j += 6)
	{
		if (b <= 10 * 1000000000ul)
		{
			const uint factor_index = atomic_inc(factor_count);
			factor[factor_index] = (ulong2)(p, b);
		}

		if (p - b <= 10 * 1000000000ul)
		{
			const uint factor_index = atomic_inc(factor_count);
			factor[factor_index] = (ulong2)(p, p - b);
		}

		b = mul_mod(b, b4, p, q);		// b = (a^k)^{j + 4}

		if (b <= 10 * 1000000000ul)
		{
			const uint factor_index = atomic_inc(factor_count);
			factor[factor_index] = (ulong2)(p, b);
		}

		if (p - b <= 10 * 1000000000ul)
		{
			const uint factor_index = atomic_inc(factor_count);
			factor[factor_index] = (ulong2)(p, p - b);
		}

		b = mul_mod(b, b2, p, q);		// b = (a^k)^{j + 6}
	}

	// ext_vector[i] = b;
}

__kernel
void clear_primes(__global uint * restrict const prime_count)
{
	*prime_count = 0;
}
