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
#include <set>

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
	ss << "ctools 24.03.0 " << sysver << ssc.str() << std::endl;
	ss << "Copyright (c) 2024, Yves Gallot" << std::endl;
	ss << "ctwin is free source code, under the MIT license." << std::endl << std::endl;
	return ss.str();
}

int expected(const int n, const uint64_t b_size, const double p_max_pos, const double p_max_neg)
{
	static const double C_p[21] = { 0, 2.2415, 4.6432, 8.0257, 7.6388, 6.1913, 6.9476, 10.2327, 10.3762, 14.1587, 14.6623, 14.5833,
		12.0591, 20.4282, 20.0690, 23.1395, 20.7106, 18.7258, 17.8171, 29.1380, 30.2934 };
	static const double C_m[21] = { 0, 3.5511, 4.4389, 4.6578, 4.6940, 4.6599, 4.6506, 4.6435, 4.6534, 4.6499, 4.6478, 4.6479,
		4.648, 4.648, 4.648, 4.648, 4.648, 4.648, 4.648, 4.648, 4.648 };
	// #candidates = (e^-gamma)^2 * C_n+ * C_n- * (b_max - b_min) / log(p_max+) / log(p_max-)
	return int(0.315236751687193398 * C_p[n] * C_m[n] * b_size / std::log(p_max_pos) / std::log(p_max_neg));
}

inline void read_cand(const int n, const uint64_t b_min, const uint64_t b_max, const double p_max_pos, const double p_max_neg, std::set<uint64_t> & cand)
{
	std::stringstream ss; ss << "data/cand_" << n << "_" << b_min << "_" << b_max << ".txt";
	const std::string filename = ss.str();

	std::cout << "Reading '" << filename << "'... " << std::flush;
	std::ifstream cand_file(filename);

	char * const buffer = new char[1 << 28];
	cand_file.rdbuf()->pubsetbuf(buffer, sizeof(buffer));

	if (cand_file.is_open())
	{
		uint64_t b = 0, b_prev = 0;
		while (cand_file.good())
		{
			cand_file >> b;
			if (b > b_prev) cand.insert(b);
			// else std::cout << "Error: " << b << std::endl;
			b_prev = b;
		}
		cand_file.close();
		const int cand_exp = expected(n, b_max - b_min, p_max_pos, p_max_neg);
		std::cout << cand.size() <<  " candidates, " << cand_exp << " expected.";
	} else std::cout << "failed.";
	std::cout << std::endl;

	delete[] buffer;
}

inline void read_factor_pos(const int n, const uint64_t b_size, const uint64_t p_min, const uint64_t p_max, const double p_max_neg, std::set<uint64_t> & cand)
{
	std::stringstream ss; ss << "data/fp" << n << "_" << p_min << "_" << p_max << ".txt";
	const std::string filename = ss.str();

	std::cout << "Reading '" << filename << "'... " << std::flush;
	std::ifstream factor_file(filename);

	char * const buffer = new char[1 << 28];
	factor_file.rdbuf()->pubsetbuf(buffer, sizeof(buffer));

	size_t count = 0;
	if (factor_file.is_open())
	{
		uint64_t p = 0, p_prev = 0, b = 0, b_prev = 0;
		while (factor_file.good())
		{
			factor_file >> p >> b;
			if ((p != p_prev) || (b != b_prev))
			{
				const MpArith mp(p);
				const uint64_t x = mp.pow(mp.toMp(b), 1 << (n - 1)), r = mp.sub(mp.mul(x, x), x);
				if (r != p - mp.one()) std::cerr << "Bad divisor: " << p << ", " << b << std::endl;
				else { cand.erase(b); ++count; }
			}
			// else std::cout << "Error: " << p << ", " << b << std::endl;
			p_prev = p; b_prev = b;
		}
		factor_file.close();
	}
	const int cand_exp = expected(n, b_size, p_max * 1e12, p_max_neg);
	std::cout << count << " factors, " << cand.size() <<  " remaining candidates, " << cand_exp << " expected." << std::endl;

	delete[] buffer;
}

inline void write_cand(const std::string & filename, const std::set<uint64_t> & cand)
{
	std::cout << "Writing '" << filename << "'... " << std::flush;

	std::ofstream out_file(filename);
	std::ostringstream buffer;
	size_t i = 0;
	for (uint64_t b : cand)
	{
		buffer << b << std::endl;
		if (++i % 1024 == 0)
		{
			out_file << buffer.str();
			buffer.str(""); buffer.clear();
		}
	}
	out_file << buffer.str();
	out_file.close();

	std::cout << cand.size() <<  " candidates." << std::endl;
}

inline void	check_prp(const int n, const std::set<uint64_t> & cand)
{
	std::stringstream ss; ss << "data/prp_" << n << ".txt";
	const std::string filename = ss.str();

	std::cout << "Checking '" << filename << "'... " << std::flush;
	std::ifstream prp_file(filename);

	size_t count = 0;
	if (prp_file.is_open())
	{
		uint64_t b = 0, b_prev = 0;
		while (prp_file.good())
		{
			prp_file >> b;
			if (b > b_prev)
			{
				if (cand.find(b) == cand.end()) std::cout << "Error: " << b << " not found." << std::endl;
				++count;
			}
			// else std::cout << "Error: " << b << std::endl;
			b_prev = b;
		}
		prp_file.close();
		std::cout << count <<  " primes.";
	} else std::cout << "failed.";
	std::cout << std::endl;
}

int main(/*int argc, char * argv[]*/)
{
	std::cout << header();

	const int n = 13;
	const uint64_t b_min = 2, b_max = 2000000000ull, b_size = b_max - b_min;
	const double p_max_pos = 561e12, p_max_neg = 1.633e12;

	std::set<uint64_t> cand;
	read_cand(n, b_min, b_max, p_max_pos, p_max_neg, cand);

	read_factor_pos(n, b_size, 1000, 2000, p_max_neg, cand);
	read_factor_pos(n, b_size, 100000, 200000, p_max_neg, cand);

	check_prp(n, cand);

	// write_cand("cand_14.txt", cand);

	return EXIT_SUCCESS;
}
