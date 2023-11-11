/*
Copyright 2023, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <iostream>
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

	// Montgomery form of 2^64 is (2^64)^2
	uint64_t two_pow_64() const
	{
		uint64_t t = add(_one, _one); t = add(t, t);	// 4
		t = add(t, t); t = add(t, t);					// 16
		for (size_t i = 0; i < 4; ++i) t = mul(t, t);	// 16^{2^4} = 2^64
		return uint64_t(t);
	}

public:
	MpArith(const uint64_t p) : _p(p), _q(invert(p)), _one((-p) % p), _r2(two_pow_64()) { }

	uint64_t toMp(const uint64_t n) const { return mul(n, _r2); }
	uint64_t toInt(const uint64_t r) const { return REDC(r); }

	uint64_t one() const { return _one; }

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

	// If p is prime and (a/p) = 1 return s such that s^2 = (a mod p), otherwise return 0
	uint64_t sqrt(const uint64_t a) const
	{
		if (_p % 4 == 3)
		{
			const uint64_t s = pow(a, (_p + 1) / 4);
			return (mul(s, s) == a) ? s : 0;
		}

		if (_p % 8 == 5)
		{
			const uint64_t b = add(a, a);
			const uint64_t v = pow(b, (_p - 5) / 8);
			const uint64_t i = mul(b, mul(v, v));	// i^2 = -1
			const uint64_t s = mul(mul(a, v), sub(i, _one));
			return (mul(s, s) == a) ? s : 0;
		}

		// Tonelli-Shanks algorithm

		// p = q * 2^e + 1, q odd
		uint64_t q = _p - 1; unsigned int e = 0; while (q % 2 == 0) { q /= 2; ++e; }

		uint64_t z = 2; while (jacobi(z, _p) != -1) ++z;
		z = pow(toMp(z), q);

		uint64_t y = z;
		unsigned int r = e;
		uint64_t x = pow(a, (q - 1) / 2);
		uint64_t b = mul(a, mul(x, x));
		x = mul(a, x);

		while (true)
		{
			if (b == _one) return x;

			unsigned int m = 1;
			uint64_t t = b; while (m < r) { t = mul(t, t); if (t == _one) break; ++m; }
			if (m == r) break;

			t = y; for (size_t i = 0; i < r - m - 1; ++i) t = mul(t, t);
			y = mul(t, t);
			r = m;
			x = mul(x, t);
			b = mul(b, y);
		}

		return 0; // n is not prime or (a/n) != 1
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
	ss << "ctsieve 0.1.0 " << sysver << ssc.str() << std::endl;
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
		std::stringstream ss; ss << "ctsieve_" << n << "_" << b_min << "_" << b_max;
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

	std::string get_sieve_filename() const { return _filename + ".sv"; }
	std::string get_cand_filename() const { return _filename + ".cand"; }
	std::string get_res_filename() const { return _filename + ".res"; }

	void info() const
	{
		static const double C_p[21] = { 0, 2.2415, 4.6432, 8.0257, 7.6388, 6.1913, 6.9476, 10.2327, 10.3762, 14.1587, 14.6623, 14.5833,
			12.0591, 20.4282, 20.0690, 23.1395, 20.7106, 18.7258, 17.8171, 29.1380, 30.2934 };
		static const double C_m[21] = { 0, 3.54, 4.43, 4.65, 4.69, 4.66, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65, 4.65 };

		// #candidates = (e^-gamma)^2 * C_n+ * C_n- * (b_max - b_min) / log(p_max+) / log(p_max-)
		const size_t size = get_size(), count = get_count();
		const size_t expected = size_t(0.315236751687193398 * C_p[_n] * C_m[_n] * (_b_max - _b_min) / std::log(_p_min_pos) / std::log(_p_min_neg));
		std::cout << "Remaining " << count << "/" << size << " candidates (" << count * 100.0 / size << "%), " << expected << " expected." << std::endl;
	}

	void read()
	{
		bool success = false;

		std::ifstream file(get_sieve_filename());
		if (file.good())
		{
			success = true;
			_bsieve.assign(_bsieve.size(), true);
			std::string line;
			if (std::getline(file, line))
			{
				std::istringstream iss(line);
				if (!(iss >> _p_min_pos) || !(iss >> _p_min_neg)) success = false;
			}
			while (std::getline(file, line))
			{
				size_t i = 0;
				try { i = std::stoull(line.c_str()); } catch (...) { success = false; }
				_bsieve[i] = false;
			}
		}
		file.close();

		if (!success)
		{
			std::ostringstream ss; ss << "Error reading file '" << get_sieve_filename() << "'.";
			throw std::runtime_error(ss.str());
		}

		info();
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

		std::ofstream file(sieve_filename);
		file << _p_min_pos << " " << _p_min_neg << std::endl;
		for (uint64_t i = 0, size = get_size(); i < size; ++i) if (!_bsieve[i]) file << i << std::endl;
		file.close();

		if (cand)
		{
			std::ofstream file(get_cand_filename());
			for (uint64_t i = 0, size = get_size(); i < size; ++i) if (!_bsieve[i]) file << _b_min + i << std::endl;
			file.close();
		}

		info();
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
			std::cout << p << "\r";
			if (std::chrono::duration<double>(now - _record_time).count() > 600)
			{
				_record_time = now;
				write(false);
			}
		}
		return !_quit;
	}

	void check_pos(const uint64_t p_max)
	{
		const int n = _n;
		const uint64_t b_min = _b_min, b_max = _b_max;

		uint64_t k_min = (_p_min_pos >> n) / 3, k_max = (p_max >> n) / 3;
		if (p_max != uint64_t(-1)) while (3 * (k_max << n) + 1 < p_max) ++k_max;

		if (!init()) return;

		std::cout << "+1: for p = " << 3 * (k_min << n) + 1 << " to " << 3 * (k_max << n) + 1 << "." << std::endl;

		for (uint64_t k = k_min; k <= k_max; ++k)
		{
			const uint64_t p = 3 * (k << n) + 1;

			if ((p % 5 == 0) || (p % 7 == 0) || (p % 11 == 0) || (p % 13 == 0) || (p % 17 == 0)
			|| (p % 19 == 0) || (p % 23 == 0) || (p % 29 == 0) || (p % 31 == 0)) continue;

			const MpArith mp(p);

			if (mp.prp())
			{
				uint64_t a = 5;
				while (true)
				{
					if ((jacobi(a, p) == -1) && (mp.pow(mp.toMp(a), k << (n - 1)) != p - mp.one())) break;
					++a; if (a % 3 == 0) ++a;
				}
				const uint64_t c = mp.pow(mp.toMp(a), k), c2 = mp.mul(c, c);

				for (uint64_t i = 1, b = mp.toInt(c); i < (uint64_t(3) << n); i += 2, b = mp.mul(b, c2))
				{
					if (i % 3 == 0) continue;

					for (uint64_t s = b; s <= b_max; s += p)
					{
						if (s >= b_min)
						{
							if (!_bsieve[s - b_min])
							{
								const Mod mod(p);
								const uint64_t x = mod.pow(s, 1 << (n - 1)), r = mod.sub(mod.mul(x, x), x);
								if (r == p - 1) _bsieve[s - b_min] = true;
								else // May fail if p is not prime
								{
									if (mod.isprime()) throw std::runtime_error("Calculation error.");
								}
							}
						}
					}
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

		std::cout << "-1: for p = " << 10 * k_min - 1 << " to " << 10 * k_max + 1 << "." << std::endl;

		for (uint64_t k = k_min; k <= k_max; ++k)
		{
			for (int i = 0; i <= 1; ++i)
			{
				const uint64_t p = 10 * k + 2 * i - 1;

				if ((p % 3 == 0) || (p % 7 == 0)) continue;
				if (p > 31)
				{
					if ((p % 11 == 0) || (p % 13 == 0) || (p % 17 == 0)	|| (p % 19 == 0)
					|| (p % 23 == 0) || (p % 29 == 0) || (p % 31 == 0)) continue;
				}

				const MpArith mp(p);

				if (mp.prp())
				{
					for (uint64_t b = b_min; b <= b_max; ++b)
					{
						if (!_bsieve[b - b_min])
						{
							const uint64_t x = mp.pow(mp.toMp(b), 1 << (n - 1));
							const uint64_t r = mp.sub(mp.mul(x, x), x);
							if (r == mp.one())
							{
								const Mod mod(p);
								const uint64_t xp = mod.pow(b, 1 << (n - 1)), rp = mod.sub(mod.mul(xp, xp), xp);
								if (rp != 1) throw std::runtime_error("Calculation error.");
								_bsieve[b - b_min] = true;
							}
						}
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
			p_max_pos = 3 * (100000000ull << _n) + 1; p_max_neg = 100001ull;
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

	// for (uint64_t k = 1; true; ++k)
	// {
	// 	const uint64_t p = 2 * k + 1;

	// 	MpArith mp(p);
	// 	if (mp.prp())
	// 	{
	// 		std::cout << p << "\r";
	// 		bool warning = true;
	// 		for (uint64_t a = 1; a < p; ++a)
	// 		{
	// 			if (jacobi(a, p) == 1)
	// 			{
	// 				const uint64_t ma = mp.toMp(a);
	// 				const uint64_t s = mp.sqrt(ma);
	// 				if (mp.mul(s, s) != ma)
	// 				{
	// 					if (s != 0)  { std::cout << "Error: s != 0" << std::endl; return 0; }
	// 					const Mod mod(p);
	// 					if (mod.isprime()) { std::cout << "Error: p = " << p << ", a = " << a << std::endl; return 0; }
	// 					else if (warning) { std::cout << "Warning: p = " << p << ", a = " << a << std::endl; warning = false; }
	// 				}
	// 			}
	// 		}
	// 	}
	// }

	// int n = 10, mode = 0;
	// uint64_t b_min = 2, b_max = 10000;

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
