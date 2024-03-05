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
	void quit() { _quit = true; }

protected:
	const size_t _factorSize = size_t(1) << 24;
	const int _log2GlobalWorkSize = 21;
	volatile bool _quit = false;
	int _n = 0;
	size_t _factorsLoop = 0;
	size_t _savedCount = 0;
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
	void initEngine(engine & engine, const int log2Global) const
	{
		const size_t globalWorkSize = size_t(1) << log2Global;

		std::stringstream src;
		src << "#define\tg_n\t" << _n << std::endl;
		src << "#define\tfactors_loop\t" << _factorsLoop << std::endl;
		src << std::endl;

		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		engine.loadProgram(src.str());
		engine.allocMemory(globalWorkSize, _factorSize);
		engine.createKernels();

		engine.clearPrimeCount();
		engine.clearFactorCount();
	}

private:
	void clearEngine(engine & engine) const
	{
		engine.releaseKernels();
		engine.releaseMemory();
		engine.clearProgram();
	}

private:
	void saveFactors(engine & engine, const double elapsedTime, const uint32_t p_min, const uint32_t p_max)
	{
		const size_t factorCount = engine.readFactorCount();

		if (factorCount == 0) return;
		if (factorCount >= _factorSize) throw std::runtime_error("factor count is too large");

		const std::string runtime = timer::formatTime(elapsedTime);
		std::cout << factorCount << " factors, time = " << runtime << std::endl;

		std::vector<cl_ulong2> factor(factorCount);
		engine.readFactors(factor.data(), factorCount);

		const std::string resFilename = std::string("f") + _extension;
		std::ofstream resFile(resFilename, std::ios::app);
		if (resFile.is_open())
		{
			const int n = _n, N = uint32_t(1) << n;
			for (size_t i = _savedCount; i < factorCount; ++i)
			{
				const cl_ulong2 & f = factor[i];
				const uint64_t p = f.s[0], b = f.s[1];
				if ((p < p_min * 1000000000000ull) || (p > p_max * 1000000000000ull)) continue;

				const Mod mod(p);
				if (mod.pow(b, 1 << n) == p - 1)
				{
					std::ostringstream ss; ss << p << " | " << b << "^" << N << "+1" << std::endl;
					resFile << ss.str();
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
						std::ostringstream ss; ss << p << " doesn't divide " << b << "^" << N << "+1";
						throw std::runtime_error(ss.str());
					}
				}
			}
			resFile.close();

			engine.clearFactorCount();
		}
	}

public:
	bool check(engine & engine, const int n, const uint32_t p_min, const uint32_t p_max)
	{
		_n = n;
		_factorsLoop = size_t(1) << std::min(_n - 1, 10);
		_savedCount = 0;
		std::stringstream ss; ss << n << "_" << p_min << "_" << p_max << ".txt";
		_extension = ss.str();

		uint64_t cnt = 0;
		const std::string ctxFilename = std::string("ctx") + _extension;
		{
			std::ifstream ctxFile(ctxFilename);
			if (ctxFile.is_open())
			{
				ctxFile >> cnt;
				ctxFile.close();
			}
		}

		initEngine(engine, _log2GlobalWorkSize);

		const double f = 1e12 / pow(2.0, double(n + 1 + _log2GlobalWorkSize));
		const uint64_t i_min = uint64_t(floor(p_min * f)), i_max = uint64_t(ceil(p_max * f));

		const size_t N_2_factors_loop = (size_t(1) << (_n - 1)) / _factorsLoop;

		const uint64_t p_min64 = ((i_min + cnt) << (_log2GlobalWorkSize + _n + 1)) + 1;
		const uint64_t p_max64 = ((((i_max << _log2GlobalWorkSize) - 1) << (_n + 1)) + 1);

		std::cout << ((cnt != 0) ? "Resuming from a checkpoint, t" : "T") << "esting n = " << _n << " from " << p_min64 << " to " << p_max64 << std::endl;

		watch chrono(0);

		const size_t globalWorkSize = size_t(1) << _log2GlobalWorkSize;

		for (uint64_t i = i_min + cnt; i < i_max; ++i)
		{
			if (_quit) break;

			engine.checkPrimes(globalWorkSize, i << _log2GlobalWorkSize);
			const size_t primeCount = engine.readPrimeCount();
			std::cout << primeCount << " primes" << std::endl;
			engine.initFactors(globalWorkSize);
			engine.checkFactors(globalWorkSize, N_2_factors_loop, 0);
			const size_t factorCount = engine.readFactorCount();
			std::cout << factorCount << " factors" << std::endl;
			engine.clearPrimes();

			++cnt;

			chrono.read();

			if (chrono.getDisplayTime() > 1)
			{
				chrono.resetDisplayTime();
				std::ostringstream ss; ss << std::setprecision(3) << " " << cnt * 100.0 / (i_max - i_min) << "% done    \r";
				std::cout << ss.str();
			}

			if (chrono.getRecordTime() > 5)
			{
				chrono.resetRecordTime();
				saveFactors(engine, chrono.getElapsedTime(), p_min, p_max);

				std::ofstream ctxFile(ctxFilename);
				if (ctxFile.is_open())
				{
					ctxFile << cnt << std::endl;
					ctxFile.close();
				}
			}
		}

		if (cnt > 0)
		{
			std::cout << " terminating...         \r";
			saveFactors(engine, chrono.getElapsedTime(), p_min, p_max);
		}

		// engine.displayProfiles(1);

		clearEngine(engine);

		return true;
	}
};
