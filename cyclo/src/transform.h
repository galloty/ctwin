/*
Copyright 2023, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include "engine.h"
#include "pio.h"
#include "file.h"

#include <cstdint>
#include <cmath>
#include <fstream>
#include <array>

#include "ocl/kernel.h"

typedef std::array<uint32, VSIZE> vint32;

class transform
{
private:
	static const uint32 P1 = 4293918721u;	// 4095 * 2^20 + 1
	static const uint32 P2 = 4253024257u;	//  507 * 2^23 + 1
	static const uint32 P3 = 4231004161u;	// 4035 * 2^20 + 1
	// static const uint32 P4 = 4221566977u;	// 2013 * 2^21 + 1

private:
	template <uint32 p, uint32 prRoot>
	class Zp
	{
	private:
		uint32 _n;

	private:
		explicit Zp(const uint32 n) : _n(n) {}

	public:
		Zp() {}
		explicit Zp(const int32_t i) : _n((i < 0) ? p - static_cast<uint32>(-i) : static_cast<uint32>(i)) {}

		static uint32 getp() { return p; }
		uint32 get() const { return _n; }
		int32_t getInt() const { return (_n > p / 2) ? static_cast<int32_t>(_n - p) : static_cast<int32_t>(_n); }

		void set(const uint32 n) { _n = n; }

		Zp operator-() const { return Zp((_n != 0) ? p - _n : 0); }

		Zp & operator+=(const Zp & rhs) { const uint32 c = (_n >= p - rhs._n) ? p : 0; _n = _n + rhs._n - c; return *this; }
		Zp & operator-=(const Zp & rhs) { const uint32 c = (_n < rhs._n) ? p : 0; _n = _n - rhs._n + c; return *this; }
		Zp & operator*=(const Zp & rhs) { _n = static_cast<uint32>((_n * uint64_t(rhs._n)) % p); return *this; }

		Zp operator+(const Zp & rhs) const { Zp r = *this; r += rhs; return r; }
		Zp operator-(const Zp & rhs) const { Zp r = *this; r -= rhs; return r; }
		Zp operator*(const Zp & rhs) const { Zp r = *this; r *= rhs; return r; }

		Zp pow(const uint32_t e) const
		{
			if (e == 0) return Zp(1);

			Zp r = Zp(1), y = *this;
			for (uint32_t i = e; i != 1; i /= 2)
			{
				if (i % 2 != 0) r *= y;
				y *= y;
			}
			r *= y;

			return r;
		}

		static Zp norm(const uint32_t n) { return -Zp((p - 1) / n); }
		static const Zp prRoot_n(const uint32_t n) { return Zp(prRoot).pow((p - 1) / n); }
	};

	// Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.

	// Montgomery form: if 0 <= n < p then r is n * 2^32 mod p
	template <uint32 p>
	class MForm
	{
	private:
		const uint32 _q;
		const uint32 _r2;	// (2^32)^2 mod p

	private:
		// p * p_inv = 1 (mod 2^32) (Newton's method)
		static constexpr uint32 invert()
		{
			uint32 p_inv = 1, prev = 0;
			while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
			return p_inv;
		}

		// The Montgomery REDC algorithm
		constexpr uint32 REDC(const uint64 t) const
		{
			const uint32 m = uint32(t) * _q, t_hi = uint32(t >> 32);
			const uint32 mp = uint32((m * uint64(p)) >> 32), r = t_hi - mp;
			return (t_hi < mp) ? r + p : r;
		}

	public:
		MForm() : _q(invert()), _r2(uint32((((uint64(1) << 32) % p) * ((uint64(1) << 32) % p)) % p)) {}

		constexpr uint32 q() const { return _q; }
		constexpr uint32 r2() const { return _r2; }

		// Conversion into Montgomery form
		constexpr uint32 toMonty(const uint32 lhs) const
		{
			// n * (2^32)^2 = (n * 2^32) * (1 * 2^32)
			return REDC(lhs * uint64(_r2));
		}

		// Conversion out of Montgomery form
		constexpr uint32 fromMonty(const uint32 lhs) const
		{
			// n = REDC(n * 2^32, 1)
			return REDC(lhs);
		}
	};

	typedef Zp<P1, 19> Zp1;
	typedef Zp<P2, 5> Zp2;
	typedef Zp<P3, 7> Zp3;
	// typedef Zp<P4, 5> Zp3;

	typedef MForm<P1> MForm1;
	typedef MForm<P2> MForm2;
	typedef MForm<P3> MForm3;

	template<class Zp1, class Zp2>
	class RNS_T
	{
	private:
		uint32_2 r;	// Zp1, Zp2

	private:
		explicit RNS_T(const Zp1 & n1, const Zp2 & n2) { r.s[0] = n1.get(); r.s[1] = n2.get(); }

	public:
		RNS_T() {}
		explicit RNS_T(const int32_t i) { r.s[0] = Zp1(i).get(); r.s[1] = Zp2(i).get(); }

		uint32_2 get() const { return r; }
		Zp1 r1() const { Zp1 n1; n1.set(r.s[0]); return n1; }
		Zp2 r2() const { Zp2 n2; n2.set(r.s[1]); return n2; }
		void set(const uint32 n1, const uint32 n2) { r.s[0] = n1; r.s[1] = n2; }

		RNS_T operator-() const { return RNS_T(-r1(), -r2()); }

		RNS_T operator+(const RNS_T & rhs) const { return RNS_T(r1() + rhs.r1(), r2() + rhs.r2()); }
		RNS_T operator-(const RNS_T & rhs) const { return RNS_T(r1() - rhs.r1(), r2() - rhs.r2()); }
		RNS_T operator*(const RNS_T & rhs) const { return RNS_T(r1() * rhs.r1(), r2() * rhs.r2()); }

		RNS_T pow(const uint32_t e) const { return RNS_T(r1().pow(e), r2().pow(e)); }

		RNS_T invert() const { return RNS_T(r1().pow(Zp1::getp() - 2), r2().pow(Zp2::getp() - 2)); }

		// Conversion into Montgomery form
		RNS_T toMonty() const
		{
			Zp1 n1; n1.set(MForm1().toMonty(r.s[0]));
			Zp2 n2; n2.set(MForm2().toMonty(r.s[1]));
			return RNS_T(n1, n2);
		}

		// Conversion out of Montgomery form
		RNS_T fromMonty() const
		{
			Zp1 n1; n1.set(MForm1().fromMonty(r.s[0]));
			Zp2 n2; n2.set(MForm2().fromMonty(r.s[1]));
			return RNS_T(n1, n2);
		}

		static const RNS_T norm(const uint32_t n) { return RNS_T(Zp1::norm(n), Zp2::norm(n)); }
		static const RNS_T prRoot_n(const uint32_t n) { return RNS_T(Zp1::prRoot_n(n), Zp2::prRoot_n(n)); }
	};

	template<class Zp3>
	class RNSe_T
	{
	private:
		uint32 r;	// Zp3

	private:
		explicit RNSe_T(const Zp3 & n3) { r = n3.get(); }

	public:
		RNSe_T() {}
		explicit RNSe_T(const int32_t i) { r = Zp3(i).get(); }

		uint32 get() const { return r; }
		Zp3 r3() const { Zp3 n3; n3.set(r); return n3; }
		void set(const uint32 n3) { r = n3; }

		RNSe_T operator-() const { return RNSe_T(-r3()); }

		RNSe_T operator+(const RNSe_T & rhs) const { return RNSe_T(r3() + rhs.r3()); }
		RNSe_T operator-(const RNSe_T & rhs) const { return RNSe_T(r3() - rhs.r3()); }
		RNSe_T operator*(const RNSe_T & rhs) const { return RNSe_T(r3() * rhs.r3()); }

		RNSe_T pow(const uint32_t e) const { return RNSe_T(r3().pow(e)); }

		RNSe_T invert() const { return RNSe_T(r3().pow(Zp3::getp() - 2)); }

		// Conversion into Montgomery form
		RNSe_T toMonty() const
		{
			Zp3 n3; n3.set(MForm3().toMonty(r));
			return RNSe_T(n3);
		}

		// Conversion out of Montgomery form
		RNSe_T fromMonty() const
		{
			Zp3 n3; n3.set(MForm3().fromMonty(r));
			return RNSe_T(n3);
		}

		static const RNSe_T norm(const uint32_t n) { return RNSe_T(Zp3::norm(n)); }
		static const RNSe_T prRoot_n(const uint32_t n) { return RNSe_T(Zp3::prRoot_n(n)); }
	};

	typedef RNS_T<Zp1, Zp2> RNS;
	typedef RNSe_T<Zp3> RNSe;

private:
	const size_t _n;
	const int32 _ln;
	const bool _isBoinc;
	engine & _engine;
	std::vector<uint32> _x;
	size_t _csize = 0;
	vint32 _b;

private:
	static size_t bitRev(const size_t i, const size_t n)
	{
		size_t r = 0;
		for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
		return r;
	}

public:
	void set(const uint32 a) const { _engine.set(a); }
	void copy(const uint32 dst, const uint32 src) const { _engine.copy(dst, src); }
	void squareDup(const uint64 dup) const { _engine.squareDup(this->_ln, dup); }
	void mul() const { _engine.mul(this->_ln); }

private:
	bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::stringstream & src) const
	{
		if (_isBoinc) return false;

		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;

		// if .cl file exists then generate header file
		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to `CRLF` 
		if (!hFile.is_open()) throw std::runtime_error("cannot write openCL header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2023, Yves Gallot" << std::endl << std::endl;
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
	void initEngine() const
	{
		const size_t n = this->_n;
		const size_t csize = this->_csize;

		std::stringstream src;
		MForm1 mf1; MForm2 mf2; MForm3 mf3;
		src << "#define\tVSIZE\t" << VSIZE << "u" << std::endl;
		src << "#define\tNSIZE\t" << n << "u" << std::endl;
		src << "#define\tLNSIZE\t" << this->_ln << std::endl;
		src << "#define\tCSIZE\t" << csize << "u" << std::endl << std::endl;
		src << "#define\tP1\t" << P1 << "u" << std::endl;
		src << "#define\tP2\t" << P2 << "u" << std::endl;
		src << "#define\tP3\t" << P3 << "u" << std::endl;
		src << "#define\tQ1\t" << mf1.q() << "u" << std::endl;
		src << "#define\tQ2\t" << mf2.q() << "u" << std::endl;
		src << "#define\tQ3\t" << mf3.q() << "u" << std::endl;
		src << "#define\tR1\t" << mf1.r2() << "u" << std::endl;
		src << "#define\tR2\t" << mf2.r2() << "u" << std::endl;
		src << "#define\tR3\t" << mf3.r2() << "u" << std::endl;
		src << "#define\tInvP2_P1\t" << mf1.toMonty(105u) << "u" << std::endl;
		src << "#define\tInvP3_P1\t" << mf1.toMonty(3220439109u) << "u" << std::endl;
		src << "#define\tInvP3_P2\t" << mf2.toMonty(607575087u) << "u" << std::endl;
		src << "#define\tP1P2P3l\t" << 8307431584695320577ull << "ul" << std::endl;
		src << "#define\tP1P2P3h\t" << 4188662890u << "u" << std::endl;
		src << "#define\tP1P2P3_2l\t" << 4153715792347660288ull << "ul" << std::endl;
		src << "#define\tP1P2P3_2h\t" << 2094331445u << "u" << std::endl;
		src << std::endl;

		// if xxx.cl file is not found then source is src_ocl_xxx string in src/ocl/xxx.h
		if (!readOpenCL("ocl/kernel.cl", "src/ocl/kernel.h", "src_ocl_kernel", src)) src << src_ocl_kernel;

		_engine.loadProgram(src.str());
		_engine.allocMemory(n, csize);
		_engine.createKernels();

		std::vector<uint32_2> wr(n), wri(n);
		std::vector<uint32> wre(n), wrie(n);

		const RNS nn_2 = RNS::norm(n / 2); const RNSe nn_2e = RNSe::norm(n / 2);

		for (size_t s = 1; s < n; s *= 2)
		{
			const RNS r_s = RNS::prRoot_n(static_cast<uint32_t>(6 * s));
			const RNSe r_se = RNSe::prRoot_n(static_cast<uint32_t>(6 * s));
			for (size_t j = 0; j < s; ++j)
			{
				const uint32_t j_s = static_cast<uint32_t>(3 * bitRev(j, s) + 1 + (2 * j) / s);
				RNS wrsj = r_s.pow(j_s); RNSe wrsje = r_se.pow(j_s);
				wri[s + s - j - 1] = RNS(-wrsj).toMonty().get(); wrie[s + s - j - 1] = RNSe(-wrsje).toMonty().get();
				if (s == 2)
				{
					// toMonty is applied twice to convert input into Montgomery form
					wrsj = wrsj.toMonty(); wrsje = wrsje.toMonty();
				}
				wr[s + j] = wrsj.toMonty().get(); wre[s + j] = wrsje.toMonty().get();
			}
			if (s == 1)
			{
				wri[0] = wr[1]; wrie[0] = wre[1];
				const RNS dsr_inv = RNS(RNS(1) - (r_s + r_s)).invert();
				const RNSe dsr_inve = RNSe(RNSe(1) - (r_se + r_se)).invert();
				// Not converted into Montgomery form such that output is converted out of Montgomery form
				wri[1] = (RNS(dsr_inv + dsr_inv) * nn_2).get();
				wrie[1] = (RNSe(dsr_inve + dsr_inve) * nn_2e).get();
			}
		}

		_engine.writeMemory_w(wr.data(), wre.data(), wri.data(), wrie.data());
	}

private:
	void releaseEngine() const
	{
		_engine.releaseKernels();
		_engine.releaseMemory();
		_engine.clearProgram();
	}

public:
	transform(const size_t size, engine & engine, const bool isBoinc, const size_t csize) :
		_n(size), _ln(ilog2_32(uint32_t(size))), _isBoinc(isBoinc), _engine(engine), _x(3 * VSIZE * size)
	{
		if (csize == 0)
		{
			std::ostringstream ss; ss << " auto-tuning..." << std::endl;
			pio::display(ss.str());

			cl_ulong bestTime = cl_ulong(-1);
			size_t bestCsize = 8;

			vint32 b; for (uint32 i = 0; i < VSIZE; ++i) b[i] = (uint32(1) << 31) + 9699690 * i;

			engine.setProfiling(true);
			for (size_t csize = 8; csize <= 64; csize *= 2)
			{
				this->_csize = csize;
				initEngine();
				set_b(b);
				set(1);
				squareDup(uint64(-1));
				engine.resetProfiles();
				const size_t m = 256;
				uint64 e = 0; for (size_t j = 0; j < VSIZE; ++j) e |= uint64(j % 2) << j;
				for (size_t i = 0; i < m; ++i) squareDup(e);
				const cl_ulong time = engine.getProfileTime();	// ns
				releaseEngine();

				std::ostringstream ss; ss << "vsize = " << VSIZE << ", csize = " << csize << ", " << time * 1e-3 / m << " us" << std::endl;
				pio::display(ss.str());

				if (time < bestTime)
				{
					bestTime = time;
					bestCsize = csize;
				}
			}

			this->_csize = bestCsize;
		}
		else
		{
			this->_csize = csize;
		}

		if (this->_csize > 4)
		{
			std::ostringstream ss; ss << "Chunk size = " << this->_csize << std::endl << std::endl;
			pio::print(ss.str());
		}

		engine.setProfiling(false);
		initEngine();
	}

public:
	virtual ~transform()
	{
		releaseEngine();
	}

public:
	size_t get_csize() const { return this->_csize; }

public:
	bool readContext(file & cFile)
	{
		const size_t num_regs = 2, size = num_regs * VSIZE * this->_n;

		std::vector<uint32_2> x12(size);
		if (!cFile.read(reinterpret_cast<char *>(x12.data()), sizeof(uint32_2) * size)) return false;
		std::vector<uint32> x3(size);
		if (!cFile.read(reinterpret_cast<char *>(x3.data()), sizeof(uint32) * size)) return false;

		_engine.writeMemory_x123(x12.data(), x3.data(), num_regs);
		return true;
	}

	void saveContext(file & cFile) const
	{
		const size_t num_regs = 2, size = num_regs * VSIZE * this->_n;

		std::vector<uint32_2> x12(size);
		std::vector<uint32> x3(size);
		_engine.readMemory_x123(x12.data(), x3.data(), num_regs);

		if (!cFile.write(reinterpret_cast<const char *>(x12.data()), sizeof(uint32_2) * size)) return;
		if (!cFile.write(reinterpret_cast<const char *>(x3.data()), sizeof(uint32) * size)) return;
	}


public:
	void set_b(const vint32 & b)
	{
		std::array<uint32_2, VSIZE> bb_inv;
		std::array<int32, VSIZE> bs;
		for (size_t i = 0; i < VSIZE; ++i)
		{
			const int s = ilog2_32(b[i]) - 1;
			this->_b[i] = b[i];
			bb_inv[i].s[0] = b[i];
			bb_inv[i].s[1] = uint32((uint64(1) << (s + 32)) / b[i]);
			bs[i] = s;
		}

		_engine.writeMemory_b(bb_inv.data(), bs.data());
	}

private:
	// f = f' * b + r such that 0 <= r < b
	static uint32 reduce(int64 & f, const uint32 b)
	{
		int64 d = 0;
		while (f < 0) { f += b; --d; }
		while (f >= b) { f -= b; ++d; }
		const uint32 r = uint32(f);
		f = d;
		return r;
	}

public:
	void getInt(const uint32 reg)
	{
		const size_t n = this->_n;
		uint32 * const x = &(this->_x.data()[reg * VSIZE * n]);
		const uint32 * const b = this->_b.data();

		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				x[k] = reg;
			}
		}

		_engine.readMemory_x3(x);

		std::array<int64, VSIZE> f; f.fill(0);

		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				RNSe xk; xk.set(x[k]);
				const int32 ixk = xk.r3().getInt();
				int64 e = f[i] + ixk;
				x[k] = reduce(e, b[i]);
				f[i] = e;
			}
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			while (f[i] != 0)
			{
				int64 e = -f[i];

				for (size_t j = 0; j < n / 2; ++j)
				{
					const size_t k = j * VSIZE + i;
					e += int32(x[k]);
					x[k] = reduce(e, b[i]);
				}

				e += f[i];

				for (size_t j = n / 2; j < n; ++j)
				{
					const size_t k = j * VSIZE + i;
					e += int32(x[k]);
					x[k] = reduce(e, b[i]);
					if (e == 0) break;
				}

				f[i] = e;
			}
		}

		for (size_t i = 0; i < VSIZE; ++i)
		{
			bool overflow = true;
			for (size_t j = n - 1; j >= n / 2; --j)
			{
				const size_t k = j * VSIZE + i;
				if (x[k] != b[i] - 1) { overflow = false; break; }
			}
			if (overflow)
			{
				overflow = false;
				for (size_t j = n / 2 - 1; j > 0; --j)
				{
					const size_t k = j * VSIZE + i;
					if (x[k] != 0) { overflow = true; break; }
				}
				if (!overflow) overflow = (x[0] != 0);
			}

			if (overflow)
			{
				int32_t carry = 0;
				for (size_t j = 0; j < n / 2; ++j)
				{
					const size_t k = j * VSIZE + i;
					const int32_t y = (j == 0) ? 1 : 0;
					const int32_t s = int32_t(x[k]) - y + carry;
					if (s < 0) { x[k] = uint32(s) + b[i]; carry = -1; } else { x[k] = uint32(s); carry = 0; break; }
				}
				if (carry != 0) pio::error("getInt failed", true);
				for (size_t j = n / 2; j < n; ++j)
				{
					const size_t k = j * VSIZE + i;
					x[k] = 0;
				}
			}
		}
	}

public:
	bool isPrime(bool * const prm, uint64_t * const res64) const
	{
		const size_t n = this->_n;
		const uint32 * const x = &(this->_x.data()[0 * VSIZE * n]);
		const uint32 * const b = this->_b.data();

		std::array<bool, VSIZE> err; err.fill(false);

		for (size_t i = 0; i < VSIZE; ++i)
		{
			prm[i] = (x[i] == 1);
			err[i] = (x[i] == 0);
		}

		std::array<uint64_t, VSIZE> bi;
		for (size_t i = 0; i < VSIZE; ++i) bi[i] = b[i];

		for (size_t i = 0; i < VSIZE; ++i) res64[i] = x[i];

		for (size_t j = 1; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				const uint32 xk = x[k];
				prm[i] &= (xk == 0);
				err[i] &= (xk == 0);
				res64[i] += xk * bi[i];
				bi[i] *= b[i];
			}
		}

		bool error = false;
		for (size_t i = 0; i < VSIZE; ++i) error |= err[i];

		return error;
	}

public:
	bool GerbiczLiCheck() const
	{
		const size_t n = this->_n;
		const uint32 * const y = &(this->_x.data()[1 * VSIZE * n]);
		const uint32 * const z = &(this->_x.data()[2 * VSIZE * n]);

		bool success = true;
		for (size_t j = 0; j < n; ++j)
		{
			for (size_t i = 0; i < VSIZE; ++i)
			{
				const size_t k = j * VSIZE + i;
				success &= (y[k] == z[k]);
			}
		}
		return success;
	}

/*public:
	void bench() const
	{
		uint32 res[8];
		_engine.bench_add_throughput(res);
		if (res[0] != 0)
		{
			const double n = 8.0 * 65536;
			const double at = res[0] / n;
			_engine.bench_add_latency(res);
			const double al = res[0] / n;
			_engine.bench_sub_throughput(res);
			const double st = res[0] / n;
			_engine.bench_sub_latency(res);
			const double sl = res[0] / n;
			_engine.bench_mul_throughput(res);
			const double mt = res[0] / n;
			_engine.bench_mul_latency(res);
			const double ml = res[0] / n;
			_engine.bench_but_throughput(res);
			const double bt = res[0] / n;
			_engine.bench_but_latency(res);
			const double bl = res[0] / n;
			std::ostringstream ss; ss << "Bench: throughput (latency)" << std::endl;
			ss << " add_mod: " << at << " (" << al << ")" << std::endl;
			ss << " sub_mod: " << st << " (" << sl << ")" << std::endl;
			ss << " mul_mod: " << mt << " (" << ml << ")" << std::endl;
			ss << " radix-2 butterfly: " << bt << " (" << bl << ")." << std::endl << std::endl;
			pio::print(ss.str());
		}
	}*/
};
