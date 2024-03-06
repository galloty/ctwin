/*
Copyright 2024, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include "ocl.h"
#include "engine.h"
#include "ctsieve.h"

#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <vector>
#if defined (_WIN32)
#include <Windows.h>
#else
#include <signal.h>
#endif

class application
{
private:
	struct deleter { void operator()(const application * const p) { delete p; } };

private:
	static void quit(int)
	{
		gfsieve::getInstance().quit();
	}

private:
#if defined (_WIN32)
	static BOOL WINAPI HandlerRoutine(DWORD) { quit(1); return TRUE; }
#endif

public:
	application()
	{
#if defined (_WIN32)	
		SetConsoleCtrlHandler(HandlerRoutine, TRUE);
#else
		signal(SIGTERM, quit);
		signal(SIGINT, quit);
#endif
	}

	virtual ~application() {}

	static application & getInstance()
	{
		static std::unique_ptr<application, deleter> pInstance(new application());
		return *pInstance;
	}

private:
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
		ss << "ctsieve 24.03.0 " << sysver << ssc.str() << std::endl;
		ss << "Copyright (c) 2024, Yves Gallot" << std::endl;
		ss << "ctwin is free source code, under the MIT license." << std::endl << std::endl;
		return ss.str();
	}

private:
	static std::string usage()
	{
		std::ostringstream ss;
		// ss << "Usage: ctsieve <n> <b_min> <b_max> <p_min> <p_max> <mode> [options]" << std::endl;
		ss << "Usage: ctsieve <n> <p_min> <p_max> <mode> [options]" << std::endl;
		ss << "  n is exponent: b^{2^n} - b^{2^{n-1}} +/- 1, 8 <= n <= 24." << std::endl;
		// ss << "  b-range is [b_min; b_max]." << std::endl;
		ss << "  p-range is [p_min; p_max], in T (10^12) values" << std::endl;
		ss << "  mode is <+> or <->:" << std::endl;
		ss << "    . +: extend the sieve limit for b^{2^n} - b^{2^{n-1}} + 1," << std::endl;
		ss << "    . -: extend the sieve limit for b^{2^n} - b^{2^{n-1}} - 1." << std::endl;
		ss << "  -d <n> or --device <n>: set device number=<n> (default 0)" << std::endl;
		return ss.str();
	}

public:
	void run(const std::vector<std::string> & args)
	{
		std::cout << header();

		platform platform;
		platform.displayDevices();

		// if (args.size() < 5) return;

		// parse args
		const int n = (args.size() > 0) ? std::atoi(args[0].c_str()) : 14;
		const uint32_t p_min = (args.size() > 1) ? uint32_t(std::atoi(args[1].c_str())) : 1000u;
		const uint32_t p_max = (args.size() > 2) ? uint32_t(std::atoi(args[2].c_str())) : 1010u;
		int d = 0;
		for (size_t i = 3, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg.substr(0, 2) == "-d")
			{
				const std::string dev = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				d = std::atoi(dev.c_str());
				if (d >= int(platform.getDeviceCount())) throw std::runtime_error("invalid device number");
			}
		}

		if ((n < 8) || (n > 24) || (p_min < 1) || (p_max <= p_min))
		{
			std::cout << usage();
			return;
		}

		gfsieve & sieve = gfsieve::getInstance();

		engine engine(platform, size_t(d));
		sieve.check(engine, n, p_min, p_max);
	}
};

int main(int argc, char * argv[])
{
	try
	{
		application & app = application::getInstance();

		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
		app.run(args);
	}
	catch (const std::runtime_error & e)
	{
		std::ostringstream ss; ss << std::endl << "error: " << e.what() << "." << std::endl;
		std::cerr << ss.str();
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
