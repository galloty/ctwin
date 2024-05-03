/*
Copyright 2023, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

template <uint32_t p, uint32_t prRoot>
class Zp
{
private:
	uint32_t _n;

private:
	explicit Zp(const uint32_t n) : _n(n) {}

public:
	Zp() {}
	explicit Zp(const int32_t i) : _n((i < 0) ? p - static_cast<uint32_t>(-i) : static_cast<uint32_t>(i)) {}

	static uint32_t getp() { return p; }
	int32_t getInt() const { return (_n > p / 2) ? static_cast<int32_t>(_n - p) : static_cast<int32_t>(_n); }

	void set(const uint32_t n) { _n = n; }

	Zp operator-() const { return Zp((_n != 0) ? p - _n : 0); }

	Zp & operator+=(const Zp & rhs) { const uint32_t c = (_n >= p - rhs._n) ? p : 0; _n = _n + rhs._n - c; return *this; }
	Zp & operator-=(const Zp & rhs) { const uint32_t c = (_n < rhs._n) ? p : 0; _n = _n - rhs._n + c; return *this; }
	Zp & operator*=(const Zp & rhs) { _n = static_cast<uint32_t>((_n * uint64_t(rhs._n)) % p); return *this; }

	Zp operator+(const Zp & rhs) const { Zp r = *this; r += rhs; return r; }
	Zp operator-(const Zp & rhs) const { Zp r = *this; r -= rhs; return r; }
	Zp operator*(const Zp & rhs) const { Zp r = *this; r *= rhs; return r; }

	Zp half() const { return Zp((_n % 2 == 0) ? _n / 2 : (_n + 1) / 2 + (p - 1) / 2); }

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

	static const Zp prRoot_n(const uint32_t n) { return Zp(prRoot).pow((p - 1) / n); }
};

typedef Zp<4095 * (1u << 20) + 1, 19> Zp1;

// P(x) mod x^n - r = a[0] + a[1].x + ... + a[n-1].x^{n-1}
typedef std::vector<Zp1> PolyMod;

constexpr size_t bitrev(const size_t i, const size_t n)
{
	size_t r = 0;
	for (size_t k = n, j = i; k > 1; k /= 2, j /= 2) r = (2 * r) | (j % 2);
	return r;
}

static void roots(Zp1 * const w, const size_t n)
{
	for (size_t s = 1; s < n; s *= 2)
	{
		const Zp1 r_s = Zp1::prRoot_n(6 * s);
		for (size_t j = 0; j < s; ++j)
		{
			const size_t j_s = 3 * bitrev(j, s) + 1 + (2 * j) / s;
			w[s + j] = r_s.pow(j_s);
		}
	}
}

static void square(Zp1 * const P, const Zp1 * const w, const size_t n)
{
	const Zp1 w_1 = w[1];
	const Zp1 d_inv = Zp1(Zp1(1) - (w_1 + w_1)).pow(Zp1::getp() - 2), d2_inv = d_inv + d_inv;

	const size_t n_2 = n / 2;

	for (size_t i = 0; i < n_2; ++i)
	{
		const Zp1 u0 = P[i + 0 * n_2], u1 = P[i + 1 * n_2], u1_w_1 = u1 * w_1;
		P[i + 0 * n_2] = u0 + u1_w_1;
		P[i + 1 * n_2] += u0 - u1_w_1;
	}

	for (size_t s = 2, m = n_2; m >= 2; s *= 2, m /= 2)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const Zp1 w_sj = w[s + j];

			for (size_t i = 0, m_2 = m / 2; i < m_2; ++i)
			{
				const size_t k = j * m + i;
				const Zp1 u0 = P[k + 0 * m_2], u1_w_sj = P[k + 1 * m_2] * w_sj;
				P[k + 0 * m_2] = u0 + u1_w_sj;
				P[k + 1 * m_2] = u0 - u1_w_sj;
			}
		}
	}

	for (size_t k = 0; k < n; ++k) P[k] *= P[k];

	for (size_t s = n_2, m = 2; m <= n_2; s /= 2, m *= 2)
	{
		for (size_t j = 0; j < s; ++j)
		{
			const Zp1 w_sj_inv = w[s + j].pow(Zp1::getp() - 2);

			for (size_t i = 0, m_2 = m / 2; i < m_2; ++i)
			{
				const size_t k = j * m + i;
				const Zp1 u0 = P[k + 0 * m_2], u1 = P[k + 1 * m_2];
				P[k + 0 * m_2] = Zp1(u0 + u1).half();
				P[k + 1 * m_2] = Zp1(u0 - u1).half() * w_sj_inv;
			}
		}
	}

	for (size_t i = 0; i < n_2; ++i)
	{
		const Zp1 u0 = P[i + 0 * n_2], u1 = P[i + 1 * n_2];
		P[i + 0 * n_2] = d2_inv * Zp1(u0 - w_1 * (u0 + u1)).half();
		P[i + 1 * n_2] = d2_inv * Zp1(u1 - u0).half();
	}
}

static void check(const size_t n, const std::string & res)
{
	Zp1 w[n]; roots(w, n);
	Zp1 P[n];

	std::cout << "Mod(1 + 2*x";
	for (size_t i = 2; i < n; ++i) std::cout << " + " << i + 1 << "*x^" << i;
	std::cout << ", x^" << n << " - x^" << n / 2 << " + 1)^2" << std::endl;
	std::cout << res << std::endl;

	for (int e = 1; e <= 4; ++e)
	{
		int32_t f = 1; for (int i = 0; i < e; ++i) f *= 10;
		for (size_t i = 0; i < n; ++i) P[i] = Zp1(int32_t(i + 1) * f);
		square(P, w, n);

		std::cout << " " << e << " ";
		for (size_t i = n - 1, j = 0; j < n; --i, ++j)
		{
			const int32_t a_i = P[i].getInt() / (f * f);
			if (j != 0) std::cout << ((a_i < 0) ? " - " : " + ");
			std::cout << std::abs(a_i);
			if (i > 0) std::cout << "*x";
			if (i > 1) std::cout << "^" << i;
		} 
		std::cout << std::endl;
	}
	std::cout << "max = " << std::setprecision(3) << std::sqrt(Zp1::getp() / n) << std::endl << std::endl;
}

int main()
{
	check(8, " = 284*x^7 + 254*x^6 + 220*x^5 + 182*x^4 - 144*x^3 - 224*x^2 - 272*x - 291");
	check(16, " = 2024*x^15 + 1916*x^14 + 1800*x^13 + 1676*x^12 + 1544*x^11 + 1404*x^10 + 1256*x^9 + 1100*x^8 - 1088*x^7 - 1408*x^6 - 1664*x^5 - 1859*x^4 - 1996*x^3 - 2078*x^2 - 2108*x - 2089");
	return EXIT_SUCCESS;
}
