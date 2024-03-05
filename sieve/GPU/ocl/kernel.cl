/*
Copyright 2024, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

typedef uint	uint32;
typedef ulong	uint64;

inline int jacobi(const uint32 x, const uint32 y)
{
	uint32 m = x, n = y;

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
		const uint32 t = n; n = m; m = t;

		m %= n;	// (m/n) = (m mod n / n)
	}

	return n;	// x and y are not coprime, return their gcd
}

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
inline uint64 two_pow_64(const uint64 p, const uint64 q, const uint64 one)
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
}

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
void check_primes(__global uint * restrict const prime_count, __global ulong3 * restrict const prime_vector, const ulong index)
{
	const uint64 k = index | get_global_id(0);

	const uint64 p = (k << (g_n + 1)) | 1, q = invert(p), one = (-p) % p;
	if (prp(p, q, one))
	{
		const uint prime_index = atomic_inc(prime_count);
		prime_vector[prime_index] = (ulong3)(p, q, one);
	}
}

__kernel
void init_factors(__global const uint * restrict const prime_count, __global const ulong3 * restrict const prime_vector,
	__global ulong2 * restrict const c_a2k_vector)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_count) return;

	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1, one = prime_vector[i].s2;

	const uint64 k = p >> (g_n + 1);

	uint32 a;
	if ((p % 3) == 2) { a = 3; }
	else
	{
		const uint32 pmod5 = p % 5;
		if ((pmod5 == 2) || (pmod5 == 3)) { a = 5; }
		else
		{
			const uint32 pmod7 = p % 7;
			if ((pmod7 == 3) || (pmod7 == 5) || (pmod7 == 6)) { a = 7; }
			else
			{
				for (a = 11; a < 256; a += 2)
				{
					const uint32 pmoda = p % a;
					if (jacobi(pmoda, a) == -1) break;
				}
				if (a >= 256) return;	// error?
			}
		}
	}
	uint64 ma = toMp(a, p, q, one);

	const uint64 c = pow_mod(ma, k, p, q);
	const uint64 a2k = mul_mod(c, c, p, q);
	c_a2k_vector[i] = (ulong2)(toInt(c, p, q), a2k);
}

__kernel
void check_factors(__global const uint * restrict const prime_count, __global const ulong3 * restrict const prime_vector,
	__global const ulong2 * restrict const c_a2k_vector, __global uint * restrict const factor_count, __global ulong2 * restrict const factor)
{
	const size_t i = get_global_id(0);
	if (i >= *prime_count) return;

	const uint64 p = prime_vector[i].s0, q = prime_vector[i].s1;
	uint64 c = c_a2k_vector[i].s0;
	const uint64 a2k = c_a2k_vector[i].s1;

	for (size_t i = 0; i < factors_loop; ++i)
	{
		if (c % 2 != 0) c = p - c;
		if (c <= 2000000000)
		{
			const uint factor_index = atomic_inc(factor_count);
			factor[factor_index] = (ulong2)(p, c);
		}
		c = mul_mod(c, a2k, p, q);		// c = a^{(2*i + 1).k}
	}

	c_a2k_vector[i].s0 = c;
}

__kernel
void clear_primes(__global uint * restrict const prime_count)
{
	*prime_count = 0;
}
