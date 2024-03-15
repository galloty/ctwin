/*
Copyright 2024, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "ocl.h"
#include "timer.h"
#include "engine.h"

#include <memory>
#include <cmath>

#include "ocl/kernel.h"

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

inline int kronecker(const uint32_t x, const uint32_t y)
{
	uint32_t m = x, n = y;

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
		std::swap(n, m);

		m %= n;	// (m/n) = ({m mod n}/n)
	}

	return 0;	// x and y are not coprime
}

class gfsieve
{
private:
	struct deleter { void operator()(const gfsieve * const p) { delete p; } };

public:
	gfsieve() {}
	virtual ~gfsieve() {}

	static gfsieve & getInstance()
	{
		static std::unique_ptr<gfsieve, deleter> pInstance(new gfsieve());
		return *pInstance;
	}

public:
	void quit()
	{
		_quit = true;
		std::cout << " terminating...         \r";
	}

protected:
	static const size_t _factorSize = size_t(1) << 24;	// * 2 * 8: 256 MB
	static const int _log2GlobalWorkSize = 21;			// * 3 * 8: 48 MB
	static const size_t _globalWorkSize = size_t(1) << _log2GlobalWorkSize;
	volatile bool _quit = false;
	int _n = 0;
	std::string _extension;

private:
	static bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::stringstream & src)
	{
		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;

		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2024, Yves Gallot" << std::endl << std::endl;
		hFile << "ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;

		hFile << "#pragma once" << std::endl << std::endl;
		hFile << "#include <cstdint>" << std::endl << std::endl;

		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"";
			for (char c : line)
			{
				if ((c == '\\') || (c == '\"')) hFile << '\\';
				hFile << c;
			}
			hFile << "\\n\" \\" << std::endl;

			src << line << std::endl;
		}
		hFile << "\"\";" << std::endl;

		hFile.close();
		clFile.close();
		return true;
	}

private:
	void initEngine(engine & engine, const int log2Global, const cl_char * const kro, const int mode) const
	{
		const size_t globalWorkSize = size_t(1) << log2Global;

		std::stringstream src;
		src << "#define\tg_n\t" << _n << std::endl;
		src << std::endl;

		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		engine.loadProgram(src.str());
		engine.allocMemory(globalWorkSize, _factorSize, mode);
		engine.createKernels(mode);

		engine.writeKro(kro);
	}

private:
	void clearEngine(engine & engine) const
	{
		engine.releaseKernels();
		engine.releaseMemory();
		engine.clearProgram();
	}

private:
	size_t saveFactors(engine & engine, const uint64_t p_min, const uint64_t p_max, const int mode)
	{
		const size_t factorCount = engine.readFactorCount();

		if (factorCount == 0) return factorCount;
		if (factorCount >= _factorSize) throw std::runtime_error("factor count is too large");

		std::vector<cl_ulong2> factor(factorCount);
		engine.readFactors(factor.data(), factorCount);

		const std::string fname = (mode == 1) ? "fp" : "fn";
		const std::string resFilename = fname + _extension;
		std::ofstream resFile(resFilename, std::ios::app);
		if (resFile.is_open())
		{
			const int n = _n;
			for (size_t i = 0; i < factorCount; ++i)
			{
				const cl_ulong2 & f = factor[i];
				const uint64_t p = f.s[0], b = f.s[1];
				if ((p < p_min) || (p > p_max)) continue;

				const Mod mod(p);
				const uint64_t x = mod.pow(b, 1u << (n - 1)), r = mod.sub(mod.mul(x, x), x);
				if (((mode > 0) && (r == p - 1)) || ((mode < 0) && (r == 1)))
				{
					resFile << p << " " << b << std::endl;
				}
				else
				{
					if (mod.pow(2, p) != 2)
					{
						std::ostringstream ss; ss << p << " is not 2-prp";
						throw std::runtime_error(ss.str());
					}
					if (mod.isprime())
					{
						const std::string sgn = (mode == 1) ? "+" : "-";
						std::ostringstream ss; ss << p << " doesn't divide " << b << "^" << (1 << n) << " - " << b << "^" << (1 << (n - 1)) << " " << sgn << " 1";
						throw std::runtime_error(ss.str());
					}
				}
			}
			resFile.close();

			engine.clearFactorCount();
		}

		 return factorCount;
	}

private:
	void saveContext(const uint64_t cnt, const double elapsedTime, const int mode) const
	{
		const std::string cname = (mode == 1) ? "ctx" : "ctx_n";
		std::ofstream ctxFile(cname + _extension);
		if (ctxFile.is_open())
		{
			ctxFile << cnt << std::endl;
			ctxFile << elapsedTime << std::endl;
			ctxFile.close();
		}
	}

private:
	void readContext(uint64_t & cnt, double & elapsedTime, const int mode) const
	{
		const std::string cname = (mode == 1) ? "ctx" : "ctx_n";
		std::ifstream ctxFile(cname + _extension);
		if (ctxFile.is_open())
		{
			ctxFile >> cnt;
			ctxFile >> elapsedTime;
			ctxFile.close();
		}
	}

public:
	bool check(engine & engine, const int n, const uint64_t p_min, const uint64_t p_max, const int mode)
	{
		_n = n;
		const uint64_t scale = (mode == 1) ? TERA : GIGA;
		std::stringstream ss; ss << n << "_" << (p_min / scale) << "_" << (p_max / scale) << ".txt";
		_extension = ss.str();

		uint64_t cnt = 0;
		double elapsedTime = 0;
		readContext(cnt, elapsedTime, mode);

		std::vector<cl_char> kro(128 * 256);
		for (uint32_t a = 5; a < 256; a += 2)
		{
			for (uint32_t k = 0; k < a; ++k) kro[(a - 5) * 128 + k] = (kronecker(k, a) == -1) ? cl_char(1) : cl_char(0);
		}

		initEngine(engine, _log2GlobalWorkSize, kro.data(), mode);
		engine.setProfiling(true);

		const double f = (mode == 1)
			? 1.0 / (3ull << (_log2GlobalWorkSize + n))
			: 1.0 / (10u << _log2GlobalWorkSize);
		const uint64_t i_min = uint64_t(floor(p_min * f)), i_max = uint64_t(ceil(p_max * f));

		const uint64_t p_min_blk = (mode == 1)
			? 3 * ((i_min + cnt) << (_log2GlobalWorkSize + _n)) + 1
			: 10 * (i_min << _log2GlobalWorkSize) - 1;
		const uint64_t p_max_blk = (mode == 1)
			? 3 * (((i_max << _log2GlobalWorkSize) - 1) << _n) + 1
			: 10 * ((i_max << _log2GlobalWorkSize) - 1) + 1;

		std::cout << ((cnt != 0) ? "Resuming from a checkpoint, t" : "T") << "esting n = " << _n << " from " << p_min_blk << " to " << p_max_blk << std::endl;

		watch chrono(elapsedTime);

		if (mode > 0) engine.clearPrimesPos();
		if (mode < 0) engine.clearPrimesNeg();

		size_t j = 0, j_sync = 10;
		for (uint64_t i = i_min + cnt; i < i_max; ++i)
		{
			if (_quit) break;

			if (mode > 0)
			{
				engine.generatePrimesPos(_globalWorkSize, i << _log2GlobalWorkSize);
				// std::cout << engine.readPrimeCount() << " primes" << std::endl;
				engine.initFactorsPos(_globalWorkSize);
				engine.generateFactorsPos(_globalWorkSize);
				// std::cout << engine.readFactorCount() << " factors" << std::endl;
				engine.clearPrimesPos();
			}
			if (mode < 0)
			{
				engine.generatePrimesNeg(_globalWorkSize, i << _log2GlobalWorkSize);
				engine.initFactorsNeg(_globalWorkSize);
				engine.generateFactorsNeg(_globalWorkSize);
				engine.clearPrimesNeg();
			}

			++cnt; ++j;

			if (j == j_sync)
			{
				const double recordTime = 120;
				const size_t factorCount = engine.readFactorCount();
				const double t = chrono.getElapsedTime(), dt = t - elapsedTime; elapsedTime = t;
				const size_t timeStepCount = size_t(std::lrint(j * recordTime / dt));
				const size_t sizeStepCount = size_t(std::lrint(j * (_factorSize / 2.0) / factorCount));
				// std::cout << "dt = " << dt << ", timeStepCount: " << timeStepCount << ", sizeStepCount: " << sizeStepCount << std::endl;

				j = 0; j_sync = std::max(std::min(timeStepCount, sizeStepCount), size_t(2));

				const size_t nf = saveFactors(engine, p_min, p_max, mode);
				saveContext(cnt, elapsedTime, mode);

				const double percent = cnt / double(i_max - i_min);
				std::ostringstream ss; ss << std::fixed << std::setprecision(3) << 100.0 * percent << "% done, "
					<< timer::formatTime(elapsedTime * (1 / percent - 1)) << " remaining, "
					<< std::lrint(nf / dt) << " factors/sec.";
				std::cout << timer::formatTime(elapsedTime) << ": " << ss.str() << std::endl;
			}
		}

		if (cnt > 0)
		{
			const size_t factorCount = engine.readFactorCount();
			if (factorCount > 0)
			{
				elapsedTime = chrono.getElapsedTime();
				saveFactors(engine, p_min, p_max, mode);
				saveContext(cnt, elapsedTime, mode);

				const double percent = cnt / double(i_max - i_min);
				std::ostringstream ss; ss << std::fixed << std::setprecision(3) << 100.0 * percent << "% done";
				if (_quit) ss << ", " << timer::formatTime(elapsedTime * (1 / percent - 1)) << " remaining";
				std::cout << timer::formatTime(elapsedTime) << ": " << ss.str() << "." << std::endl;
			}
		}

		engine.displayProfiles(1);
		clearEngine(engine);

		return true;
	}
};
