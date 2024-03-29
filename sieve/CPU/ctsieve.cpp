/*
Copyright 2023, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <sys/stat.h>
#if defined(_WIN32)
#include <Windows.h>
#else
#include <signal.h>
#endif

inline int jacobi(const uint64_t x, const uint64_t y)
{
	static int jac2[2 * 2] = { 1, -1, -1, 1 };
	static int jac3[2 * 3] = { 1, 0, -1, -1, 0, 1 };
	static int jac5[1 * 5] = { 1, -1, 0, -1, 1 };
	static int jac6[2 * 6] = { 1, 0, 1, -1, 0, -1, -1, 0, -1, 1, 0, 1 };
	static int jac7[2 * 7] = { 1, 1, -1, 0, 1, -1, -1, -1, -1, 1, 0, -1, 1, 1 };
	static int jac8[8 / 2] = { 1, -1, -1, 1 };
	static int jac9[1 * 9] = { 1, 0, 1, 1, 0, 1, 1, 0, 1 };

	const uint64_t y_2 = y / 2;
	if (x == 2) return jac2[y_2 % (2 * 2)];
	if (x == 3) return jac3[y_2 % (2 * 3)];
	if (x == 4) return 1;
	if (x == 5) return jac5[y_2 % (1 * 5)];
	if (x == 6) return jac6[y_2 % (2 * 6)];
	if (x == 7) return jac7[y_2 % (2 * 7)];
	if (x == 8) return jac8[y_2 % (8 / 2)];
	if (x == 9) return jac9[y_2 % (1 * 9)];

	uint64_t m = x, n = y;

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
		const uint64_t t = n; n = m; m = t;

		m %= n;	// (m/n) = (m mod n / n)
	}

	return 0;	// x and y are not coprime
}

// Modular operations, slow implementation
class Mod
{
private:
	const uint64_t _n;

private:
	static constexpr int ilog2(const uint64_t x) { return 63 - __builtin_clzll(x); }

public:
	Mod(const uint64_t n) : _n(n) { }

	uint64_t add(const uint64_t a, const uint64_t b) const
	{
		const uint64_t c = (a >= _n - b) ? _n : 0;
		return a + b - c;
	}

	uint64_t sub(const uint64_t a, const uint64_t b) const
	{
		const uint64_t c = (a < b) ? _n : 0;
		return a - b + c;
	}

	uint64_t mul(const uint64_t a, const uint64_t b) const
	{
		return uint64_t((a * __uint128_t(b)) % _n);
	}

	// a^e mod n, left-to-right algorithm
	uint64_t pow(const uint64_t a, const uint64_t e) const
	{
		uint64_t r = a;
		for (int b = ilog2(e) - 1; b >= 0; --b)
		{
			r = mul(r, r);
			if ((e & (uint64_t(1) << b)) != 0) r = mul(r, a);
		}
		return r;
	}

	bool spsp(const uint64_t p) const
	{
		// n - 1 = 2^k * r
		uint64_t r = _n - 1;
		int k = 0;
		for (; r % 2 == 0; r /= 2) ++k;

		uint64_t x = pow(p, r);
		if (x == 1) return true;

		// Compute x^(2^i) for 0 <= i < n.  If any are -1, n is a p-spsp.
		for (; k > 0; --k)
		{
			if (x == _n - 1) return true;
			x = mul(x, x);
		}

		return false;
	}

	bool isprime() const
	{
		if (_n < 2) return false;
		if (_n % 2 == 0) return (_n == 2);
		if (_n < 9) return true;

		// see https://oeis.org/A014233

		if (!spsp(2)) return false;
		if (_n < 2047ull) return true;

		if (!spsp(3)) return false;
		if (_n < 1373653ull) return true;

		if (!spsp(5)) return false;
		if (_n < 25326001ull) return true;

		if (!spsp(7)) return false;
		if (_n < 3215031751ull) return true;

		if (!spsp(11)) return false;
		if (_n < 2152302898747ull) return true;

		if (!spsp(13)) return false;
		if (_n < 3474749660383ull) return true;

		if (!spsp(17)) return false;
		if (_n < 341550071728321ull) return true;

		if (!spsp(19)) return false;
		// if (_n < 341550071728321ull) return true;

		if (!spsp(23)) return false;
		if (_n < 3825123056546413051ull) return true;

		if (!spsp(29)) return false;
		// if (_n < 3825123056546413051ull) return true;

		if (!spsp(31)) return false;
		// if (_n < 3825123056546413051ull) return true;

		if (!spsp(37)) return false;
		return true;	// 318665857834031151167461
	}
};

// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.
class MpArith
{
private:
	const uint64_t _p, _q;
	const uint64_t _one;	// 2^64 mod p
	const uint64_t _r2;		// (2^64)^2 mod p

private:
	static constexpr int ilog2(const uint64_t x) { return 63 - __builtin_clzll(x); }
	static constexpr uint64_t mul_hi(const uint64_t lhs, const uint64_t rhs) { return uint64_t((lhs * __uint128_t(rhs)) >> 64); }

	// p * p_inv = 1 (mod 2^64) (Newton's method)
	static uint64_t invert(const uint64_t p)
	{
		uint64_t p_inv = 1, prev = 0;
		while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
		return p_inv;
	}

	uint64_t REDC(const __uint128_t t) const
	{
		const uint64_t mp = mul_hi(uint64_t(t) * _q, _p), t_hi = uint64_t(t >> 64), r = t_hi - mp;
		return (t_hi < mp) ? r + _p : r;
	}

	uint64_t REDCshort(const uint64_t t) const
	{
		const uint64_t mp = mul_hi(t * _q, _p);
		return (mp != 0) ? _p - mp : 0;
	}

	// Montgomery form of 2^64 is (2^64)^2
	uint64_t two_pow_64() const
	{
		uint64_t t = add(_one, _one); t = add(t, t);	// 4
		t = add(t, t); t = add(t, t);					// 16
		for (size_t i = 0; i < 4; ++i) t = mul(t, t);	// 16^{2^4} = 2^64
		return t;
	}

public:
	MpArith(const uint64_t p) : _p(p), _q(invert(p)), _one((-p) % p), _r2(two_pow_64()) { }

	uint64_t toMp(const uint64_t n) const { return mul(n, _r2); }
	uint64_t toInt(const uint64_t r) const { return REDCshort(r); }

	uint64_t one() const { return _one; }
	uint64_t two() const { return add(_one, _one); }
	uint64_t three() const { return add(two(), _one); }
	uint64_t four() const { const uint64_t t = two(); return add(t, t); }
	uint64_t five() const { return add(four(), _one); }

	uint64_t neg(const uint64_t a) const { return (a != 0) ? _p - a : 0; }

	uint64_t add(const uint64_t a, const uint64_t b) const
	{
		const uint64_t c = (a >= _p - b) ? _p : 0;
		return a + b - c;
	}

	uint64_t sub(const uint64_t a, const uint64_t b) const
	{
		const uint64_t c = (a < b) ? _p : 0;
		return a - b + c;
	}

	uint64_t half(const uint64_t a) const
	{
		return (a % 2 == 0) ? a / 2 : a / 2 + _p / 2 + 1;
	}

	uint64_t mul(const uint64_t a, const uint64_t b) const
	{
		return REDC(a * __uint128_t(b));
	}

	// a^e mod p, left-to-right algorithm
	uint64_t pow(const uint64_t a, const uint64_t e) const
	{
		uint64_t r = a;
		for (int b = ilog2(e) - 1; b >= 0; --b)
		{
			r = mul(r, r);
			if ((e & (uint64_t(1) << b)) != 0) r = mul(r, a);
		}
		return r;
	}

	// If p is prime and (a/p) = 1 return s such that s^2 = (a mod p), otherwise return 0
	uint64_t sqrt(const uint64_t a) const
	{
		if ((a == 0) || (a == _one)) return a;

		if (_p % 4 == 3) return pow(a, (_p + 1) / 4);

		if (_p % 8 == 5)
		{
			const uint64_t b = add(a, a);
			const uint64_t v = pow(b, (_p - 5) / 8);
			const uint64_t i = mul(b, mul(v, v));	// i^2 = -1
			return mul(mul(a, v), sub(i, _one));
		}

		// Tonelli-Shanks algorithm

		// p = q * 2^e + 1, q odd
		uint64_t q = _p - 1; unsigned int e = 0; while (q % 2 == 0) { q /= 2; ++e; }

		uint64_t z = 3; while (jacobi(z, _p) != -1) ++z;

		z = pow(toMp(z), q);

		uint64_t y = z;
		unsigned int r = e;
		uint64_t x = pow(a, (q - 1) / 2);
		uint64_t b = mul(a, mul(x, x));
		x = mul(a, x);

		for (size_t j = 0; j < 64; ++j)
		{
			if (b == _one) return x;

			unsigned int m = 1;
			uint64_t t = b; while (m < r) { t = mul(t, t); if (t == _one) break; ++m; }
			if (m == r) break;

			t = y; for (size_t i = 0; i < r - 1 - m; ++i) t = mul(t, t);
			y = mul(t, t);
			r = m;
			x = mul(x, t);
			b = mul(b, y);
		}

		return 0; // n is not prime or (a/n) != 1
	}

	uint64_t sqrt_checked(const uint64_t a) const
	{
		const uint64_t s = sqrt(a);
		if (mul(s, s) != a)
		{
			if (Mod(_p).isprime())
			{
				std::ostringstream ss; ss << "Calculation error (sqrt): p = " << _p << ", a = " << toInt(a) << ".";
				throw std::runtime_error(ss.str());
			}
			return 0;
		}
		return s;
	}

	void sqrtn(const uint64_t a, const int n, std::vector<uint64_t> & L) const
	{
		// (a/p) = 1 here
		// if (jacobi(toInt(a), _p) != 1) return;

		int e = 1; for (uint64_t q = (_p - 1) / 2; q % 2 == 0; q /= 2) e++;
		e = std::min(e, n);
		if ((e > 1) && pow(a, (_p - 1) >> e) != _one) return;

		uint64_t r = a, q = (_p - 1) / 2, m = 1;
		for (int i = 1; i <= n; ++i)
		{
			if (q % 2 == 0)
			{
				r = sqrt_checked(r);
				q /= 2;
				if (r == 0) break;
			}
			else
			{
				if (m % 2 != 0) m += q;
				m /= 2;
			}
		}
		if (m > 1) r = pow(r, m);

		if (r == 0)
		{
			if (Mod(_p).isprime())
			{
				std::ostringstream ss; ss << "Calculation error (sqrtn): p = " << _p << ", a = " << toInt(a) << ", n = " << n << ".";
				throw std::runtime_error(ss.str());
			}
			return;
		}

		// u is a primitive (2^e)th root of unity
		uint64_t u = 2;
		while (true)
		{
			if (jacobi(u, _p) == -1) break;
			++u; if ((u == 4) || (u == 9)) ++u;
		}
		u = pow(toMp(u), (_p - 1) >> e);

		const size_t s = size_t(1) << e;
		L.reserve(s);
		for (size_t i = 0; i < s; i += 2)
		{
			const uint64_t b = toInt(r);
			L.push_back(b); L.push_back(_p - b);
			r = mul(r, u);
		}
	}

	// 2^(p - 1) ?= 1 mod p
	bool prp() const
	{
		const uint64_t e = (_p - 1) / 2;
		int b = ilog2(e) - 1;
		uint64_t r = add(_one, _one); r = add(r, r);
		if ((e & (uint64_t(1) << b)) != 0) r = add(r, r);
		for (--b; b >= 0; --b)
		{
			r = mul(r, r);
			if ((e & (uint64_t(1) << b)) != 0) r = add(r, r);
		}
		return ((r == _one) || (r == _p - _one));
	}
};

static std::string header()
{
	const char * const sysver =
#if defined(_WIN64)
		"win64";
#elif defined(_WIN32)
		"win32";
#elif defined(__linux__)
#ifdef __x86_64
		"linux64";
#else
		"linux32";
#endif
#elif defined(__APPLE__)
		"macOS";
#else
		"unknown";
#endif

	std::ostringstream ssc;
#if defined(__GNUC__)
	ssc << " gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#elif defined(__clang__)
	ssc << " clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#endif

	std::ostringstream ss;
	ss << "ctsieve 23.11.0 " << sysver << ssc.str() << std::endl;
	ss << "Copyright (c) 2023, Yves Gallot" << std::endl;
	ss << "ctwin is free source code, under the MIT license." << std::endl << std::endl;
	return ss.str();
}

static std::string usage()
{
	std::ostringstream ss;
	ss << "Usage: ctsieve <n> <b_min> <b_max> <mode>" << std::endl;
	ss << "  n is exponent: b^{2^n} - b^{2^{n-1}} +/- 1, 8 <= n <= 24." << std::endl;
	ss << "  range is [b_min; b_max]." << std::endl;
	ss << "  mode is in <ini> or <+> or <->:" << std::endl;
	ss << "    . ini: create a new list of candidates," << std::endl;
	ss << "    . +: extend the sieve limit for b^{2^n} - b^{2^{n-1}} + 1," << std::endl;
	ss << "    . -: extend the sieve limit for b^{2^n} - b^{2^{n-1}} - 1." << std::endl;
	return ss.str();
}

inline std::string uint2string(const uint64_t n)
{
	double f = double(n); std::string prefix;
	if (f >= 1e15) { f *= 1e-15; prefix = "P"; }
	else if (f >= 1e12) { f *= 1e-12; prefix = "T"; }
	else if (f >= 1e9) { f *= 1e-9; prefix = "G"; }
	else if (f >= 1e6) { f *= 1e-6; prefix = "M"; }
	else return std::to_string(n);
	return std::to_string(f).substr(0, 5) + prefix;
}

class Sieve
{
private:
	const int _n;
	const uint64_t _b_min, _b_max;
	std::vector<bool> _bsieve;
	uint64_t _p_min_pos, _p_min_neg;
	inline static volatile bool _quit = false;
	std::string _filename;
	std::chrono::high_resolution_clock::time_point _display_time, _record_time;

private:
	static void quit(int) { _quit = true; }

#if defined(_WIN32)
	static BOOL WINAPI HandlerRoutine(DWORD) { quit(1); return TRUE; }
#endif

public:
	Sieve(const int n, const uint64_t b_min, const uint64_t b_max)
		: _n(n), _b_min(b_min), _b_max(b_max), _bsieve(b_max - b_min + 1, false)
	{
		_p_min_pos = _p_min_neg = 0;
		std::stringstream ss; ss << "_" << n << "_" << b_min << "_" << b_max;
		_filename = ss.str();

#if defined(_WIN32)
		SetConsoleCtrlHandler(HandlerRoutine, TRUE);
#else
		signal(SIGTERM, quit);
		signal(SIGINT, quit);
#endif
	}

	virtual ~Sieve() {}

private:
	size_t get_size() const { return _bsieve.size(); }
	size_t get_count() const { size_t count = 0; for (bool b : _bsieve) if (!b) count++; return count; }

	std::string get_sieve_filename() const { return "sv" + _filename + ".dat"; }
	std::string get_cand_filename() const { return "cand" + _filename + ".txt"; }

	// Rosetta Code, CRC-32, C
	static uint32_t rc_crc32(const uint32_t crc32, const char * const buf, const size_t len)
	{
		static uint32_t table[256];
		static bool have_table = false;
	
		// This check is not thread safe; there is no mutex
		if (!have_table)
		{
			// Calculate CRC table
			for (size_t i = 0; i < 256; ++i)
			{
				uint32_t rem = static_cast<uint32_t>(i);  // remainder from polynomial division
				for (size_t j = 0; j < 8; ++j)
				{
					if (rem & 1)
					{
						rem >>= 1;
						rem ^= 0xedb88320;
					}
					else rem >>= 1;
				}
				table[i] = rem;
			}
			have_table = true;
		}

		uint32_t crc = ~crc32;
		for (size_t i = 0; i < len; ++i)
		{
			const uint8_t octet = static_cast<uint8_t>(buf[i]);  // Cast to unsigned octet
			crc = (crc >> 8) ^ table[(crc & 0xff) ^ octet];
		}
		return ~crc;
	}

	void info(const bool extended) const
	{
		static const double C_p[21] = { 0, 2.2415, 4.6432, 8.0257, 7.6388, 6.1913, 6.9476, 10.2327, 10.3762, 14.1587, 14.6623, 14.5833,
			12.0591, 20.4282, 20.0690, 23.1395, 20.7106, 18.7258, 17.8171, 29.1380, 30.2934 };
		static const double C_m[21] = { 0, 3.5511, 4.4389, 4.6578, 4.6940, 4.6599, 4.6506, 4.6435, 4.6534, 4.6499, 4.6478, 4.6479,
			4.648, 4.648, 4.648, 4.648, 4.648, 4.648, 4.648, 4.648, 4.648 };
		// #candidates = (e^-gamma)^2 * C_n+ * C_n- * (b_max - b_min) / log(p_max+) / log(p_max-)
		const size_t size = get_size(), count = get_count();
		const size_t expected = size_t(0.315236751687193398 * C_p[_n] * C_m[_n] * (_b_max - _b_min) / std::log(_p_min_pos) / std::log(_p_min_neg));
		std::cout << "Remaining " << count << "/" << size << " candidates (" << count * 100.0 / size << "%)." << std::endl;
		if (extended)
		{
			std::cout << "p_+ = " << _p_min_pos << " (" << uint2string(_p_min_pos) << "), p_- = " << _p_min_neg
				<< " (" << uint2string(_p_min_neg) << "), expected: " << expected * 100.0 / size << "%." << std::endl;
		}
	}

	void read()
	{
		bool success = false;
		_bsieve.assign(_bsieve.size(), true);

		std::ifstream file(get_sieve_filename(), std::ios::binary);
		if (file.good())
		{
			file.read(reinterpret_cast<char *>(&_p_min_pos), sizeof(_p_min_pos));
			file.read(reinterpret_cast<char *>(&_p_min_neg), sizeof(_p_min_neg));
			uint64_t sieve_size;
			file.read(reinterpret_cast<char *>(&sieve_size), sizeof(sieve_size));
			std::vector<uint32_t> sieve(sieve_size);
			file.read(reinterpret_cast<char *>(sieve.data()), static_cast<std::streamsize>(sieve_size * sizeof(uint32_t)));
			uint32_t crc32f = 0;
			file.read(reinterpret_cast<char *>(&crc32f), sizeof(crc32f));

			uint32_t crc32 = 0;
			crc32 = rc_crc32(crc32, reinterpret_cast<const char *>(&_p_min_pos), sizeof(_p_min_pos));
			crc32 = rc_crc32(crc32, reinterpret_cast<const char *>(&_p_min_neg), sizeof(_p_min_neg));
			crc32 = rc_crc32(crc32, reinterpret_cast<const char *>(&sieve_size), sizeof(sieve_size));
			crc32 = rc_crc32(crc32, reinterpret_cast<const char *>(sieve.data()), sieve_size * sizeof(uint32_t));
			crc32 = ~crc32 ^ 0xa23777ac;

			if (file && (crc32 == crc32f))
			{
				success = true;
				for (const uint32_t i : sieve) _bsieve[i] = false;
			}
		}
		file.close();

		if (!success)
		{
			std::ostringstream ss; ss << "Error reading file '" << get_sieve_filename() << "'.";
			throw std::runtime_error(ss.str());
		}

		info(true);
	}

	void write(const bool cand) const
	{
		struct stat s;
		const std::string sieve_filename = get_sieve_filename(), old_filename = sieve_filename + ".old";
		std::remove(old_filename.c_str());
		if ((stat(sieve_filename.c_str(), &s) == 0) && (std::rename(sieve_filename.c_str(), old_filename.c_str()) != 0))	// file exists and cannot rename it
		{
			std::ostringstream ss; ss << "Error writing file '" << sieve_filename << "'.";
			throw std::runtime_error(ss.str());
		}

		std::vector<uint32_t> sieve;
		for (size_t i = 0, size = _bsieve.size(); i < size; ++i) if (!_bsieve[i]) sieve.push_back(uint32_t(i));
		const uint64_t sieve_size = sieve.size();

		uint32_t crc32 = 0;
		crc32 = rc_crc32(crc32, reinterpret_cast<const char *>(&_p_min_pos), sizeof(_p_min_pos));
		crc32 = rc_crc32(crc32, reinterpret_cast<const char *>(&_p_min_neg), sizeof(_p_min_neg));
		crc32 = rc_crc32(crc32, reinterpret_cast<const char *>(&sieve_size), sizeof(sieve_size));
		crc32 = rc_crc32(crc32, reinterpret_cast<const char *>(sieve.data()), sieve_size * sizeof(uint32_t));
		crc32 = ~crc32 ^ 0xa23777ac;

		std::ofstream file(sieve_filename, std::ios::binary);
		file.write(reinterpret_cast<const char *>(&_p_min_pos), sizeof(_p_min_pos));
		file.write(reinterpret_cast<const char *>(&_p_min_neg), sizeof(_p_min_neg));
		file.write(reinterpret_cast<const char *>(&sieve_size), sizeof(sieve_size));
		file.write(reinterpret_cast<const char *>(sieve.data()), static_cast<std::streamsize>(sieve.size() * sizeof(uint32_t)));
		file.write(reinterpret_cast<const char *>(&crc32), sizeof(crc32));
		file.close();

		if (cand)
		{
			std::ofstream file(get_cand_filename());
			for (size_t i = 0, size = get_size(); i < size; ++i) if (!_bsieve[i]) file << _b_min + i << std::endl;
			file.close();
		}

		info(cand);
	}

	bool init()
	{
		_display_time = _record_time = std::chrono::high_resolution_clock::now();
		return !_quit;
	}

	bool monitor(const uint64_t p)
	{
		auto now = std::chrono::high_resolution_clock::now();
		if (std::chrono::duration<double>(now - _display_time).count() > 1)
		{
			_display_time = now;
			std::cout << uint2string(p) << "\r";
			if (std::chrono::duration<double>(now - _record_time).count() > 30 * 60)
			{
				_record_time = now;
				write(false);
			}
		}
		return !_quit;
	}

	bool check_root(const uint64_t b, const uint64_t b_min, const uint64_t b_max, const uint64_t p, const int n, const bool is_pos)
	{
		const uint64_t res = is_pos ? (p - 1) : 1;
		for (uint64_t s = b; s <= b_max; s += p)
		{
			if (s >= b_min)
			{
				if (!_bsieve[s - b_min])
				{
					const Mod mod(p);
					const uint64_t x = mod.pow(s, 1 << (n - 1)), r = mod.sub(mod.mul(x, x), x);
					if (r == res) _bsieve[s - b_min] = true;
					// May fail if p is not prime
					else if (mod.isprime())
					{
						std::ostringstream ss; ss << "Calculation error (check_" << (is_pos ? "pos" : "neg") << "): p = " << p << ", b = " << s << ".";
						throw std::runtime_error(ss.str());
					}
					else return false;
				}
			}
		}
		return true;
	}

	void check_pos(const uint64_t p_max)
	{
		const int n = _n;
		const uint64_t b_min = _b_min, b_max = _b_max;

		uint64_t k_min = (_p_min_pos >> n) / 3, k_max = (p_max >> n) / 3;
		if (p_max != uint64_t(-1)) while (3 * (k_max << n) + 1 < p_max) ++k_max;

		if (!init()) return;

		std::cout << "+1: for p = " << uint2string(3 * (k_min << n) + 1) << " to " << uint2string(3 * (k_max << n) + 1) << std::endl;

		for (uint64_t k = k_min; k <= k_max; ++k)
		{
			const uint64_t p = 3 * (k << n) + 1;

			if ((p % 5 == 0) || (p % 7 == 0) || (p % 11 == 0) || (p % 13 == 0) || (p % 17 == 0)
			 || (p % 19 == 0) || (p % 23 == 0) || (p % 29 == 0) || (p % 31 == 0)) continue;

			const MpArith mp(p);

			if (mp.prp())
			{
				uint64_t a = 5, ma = mp.five();
				while (true)
				{
					if ((jacobi(a, p) == -1) && (mp.pow(ma, k << (n - 1)) != p - mp.one())) break;
					++a; ma = mp.add(ma, mp.one());
					if (a % 3 == 0) { ++a; ma = mp.add(ma, mp.one()); }
				}

				const uint64_t c = mp.pow(ma, k), b2 = mp.mul(c, c), b4 = mp.mul(b2, b2);
				uint64_t b = mp.toInt(c);

				for (uint64_t i = 1; i < (uint64_t(3) << (n - 1)); i += 6)
				{
					if (!check_root(b, b_min, b_max, p, n, true)) break;
					if (!check_root(p - b, b_min, b_max, p, n, true)) break;
					b = mp.mul(b, b4);
					if (!check_root(b, b_min, b_max, p, n, true)) break;
					if (!check_root(p - b, b_min, b_max, p, n, true)) break;
					b = mp.mul(b, b2);
				}

				_p_min_pos = p;
				if (!monitor(p)) return;
			}
		}
	}

	void check_neg(const uint64_t p_max)
	{
		const int n = _n;
		const uint64_t b_min = _b_min, b_max = _b_max;

		uint64_t k_min = _p_min_neg / 10, k_max = p_max / 10;
		if (p_max != uint64_t(-1)) while (10 * k_max + 1 < p_max) ++k_max;

		if (!init()) return;

		std::cout << "-1: for p = " << uint2string(std::max(10 * k_min - 1, uint64_t(11))) << " to " << uint2string(10 * k_max + 1) << std::endl;

		for (uint64_t k = k_min; k <= k_max; ++k)
		{
			for (uint64_t i = 0; i <= 1; ++i)
			{
				const uint64_t p = 10 * k + 2 * i - 1;

				if ((p % 3 == 0) || (p % 7 == 0)) continue;
				if (p > 31)
				{
					if ((p % 11 == 0) || (p % 13 == 0) || (p % 17 == 0)	|| (p % 19 == 0)
					 || (p % 23 == 0) || (p % 29 == 0) || (p % 31 == 0)) continue;
				}

				const MpArith mp(p);

				if (mp.prp() && (p != 1093*1093) && (p != 3511*3511))
				{
					// std::cout << p << "\r";

					std::vector<uint64_t> L;
					const uint64_t s5 = mp.sqrt_checked(mp.toMp(5));
					if (s5 != 0)
					{
						const uint64_t r = mp.half(mp.sub(mp.one(), s5));
						if (p % 4 == 3)
						{
							const uint64_t rs = (jacobi(mp.toInt(r), p) != 1) ? mp.sub(mp.one(), r) : r;
							mp.sqrtn(rs, n - 1, L);
						}
						else if (jacobi(mp.toInt(r), p) == 1)
						{
							std::vector<uint64_t> L1; mp.sqrtn(r, n - 1, L1);
							std::vector<uint64_t> L2; mp.sqrtn(mp.sub(mp.one(), r), n - 1, L2);
							L.reserve(L1.size() + L2.size()); L.insert(L.end(), L1.begin(), L1.end()); L.insert(L.end(), L2.begin(), L2.end());
						}
					}

					for (const uint64_t & b : L)
					{
						if (!check_root(b, b_min, b_max, p, n, false)) break;
					}

					_p_min_neg = p;
					if (!monitor(p)) return;
				}
			}
		}
	}

public:
	void check(const int mode)
	{
		uint64_t p_max_pos, p_max_neg;
		if (mode != 0)
		{
			read();
			p_max_pos = p_max_neg = uint64_t(-1);
		}
		else
		{
			_p_min_pos = 3 * (1ull << _n) + 1; _p_min_neg = 11;
			p_max_pos = 3 * (100000000ull << _n) + 1; p_max_neg = 10000000001ull;
			// p_max_pos = 3 * (10000000ull << _n) + 1; p_max_neg = 1000000001ull;
		}

		std::cout << "ctwin-" << _n << ": b in [" << _b_min << ", " << _b_max << "]." << std::endl;

		if (mode >= 0) check_pos(p_max_pos);
		if (mode <= 0) check_neg(p_max_neg);

		write(true);
	}
};

int main(int argc, char * argv[])
{
	std::cout << header();
	std::cout << std::fixed << std::setprecision(3);

	// int n = 10, mode = 0;
	// uint64_t b_min = 2, b_max = 1000000;

	if (argc != 5)
	{
		std::cout << usage();
		return EXIT_SUCCESS;
	}

	int n = 0, mode = 2;
	uint64_t b_min = 0, b_max = 0;
	try
	{
		n = std::atoi(argv[1]);
		b_min = std::stoull(argv[2]);
		b_max = std::stoull(argv[3]);
		const std::string arg4(argv[4]);
		if (arg4 == "ini") mode = 0;
		else if (arg4 == "+") mode = 1;
		else if (arg4 == "-") mode = -1;
	}
	catch (...)
	{
		std::cout << usage();
		return EXIT_SUCCESS;
	}

	if ((n < 8) || (n > 24)) { std::cerr << "n must be in [8, 24]." << std::endl; return EXIT_FAILURE; }
	if (b_min < 2) b_min = 2;
	if (b_max < b_min) b_max = b_min;
	if ((mode < -1) || (mode > 1)) { std::cerr << "mode must be <ini> or <+> or <->." << std::endl; return EXIT_FAILURE; }

	Sieve sieve(n, b_min, b_max);

	try
	{
		sieve.check(mode);
	}
	catch (const std::runtime_error & e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
