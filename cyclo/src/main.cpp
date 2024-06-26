/*
Copyright 2023, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include "pio.h"
#include "ocl.h"
#include "cyclo.h"
#include "boinc.h"

#if defined(BOINC)
#include "version.h"
#endif

#include <cstdlib>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
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
		cyclo::getInstance().quit();
	}

private:
#if defined(_WIN32)
	static BOOL WINAPI HandlerRoutine(DWORD)
	{
		quit(1);
		return TRUE;
	}
#endif

public:
	application()
	{
#if defined(_WIN32)	
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
	static std::string header(const std::vector<std::string> & args, const bool nl = false)
	{
		const char * const sysver =
#if defined(_WIN64)
			"win64";
#elif defined(_WIN32)
			"win32";
#elif defined(__linux__)
#if defined(__x86_64)
			"linux x64";
#elif defined(__aarch64__)
			"linux arm64";
#else
			"linux x86";
#endif
#elif defined(__APPLE__)
#if defined(__aarch64__)
			"macOS arm64";
#else
			"macOS x64";
#endif
#else
			"unknown";
#endif

		std::ostringstream ssc;
#if defined(__clang__)
		ssc << ", clang-" << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
		ssc << ", gcc-" << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#endif

#if defined(BOINC)
		ssc << ", boinc-" << BOINC_VERSION_STRING;
#endif

		std::ostringstream ss;
		ss << "cyclo 24.4.1 (" << sysver << ssc.str() << ")" << std::endl;
		ss << "Copyright (c) 2023, Yves Gallot" << std::endl;
		ss << "ctwin is free source code, under the MIT license." << std::endl;
		if (nl)
		{
			ss << std::endl << "Command line: '";
			bool first = true;
			for (const std::string & arg : args)
			{
				if (first) first = false; else ss << " ";
				ss << arg;
			}
			ss << "'" << std::endl << std::endl;
		}
		return ss.str();
	}

private:
	static std::string usage()
	{
		std::ostringstream ss;
		ss << "Usage: cyclo [options]  options may be specified in any order" << std::endl;
		ss << "  -n <n>                  exponent (b^{2^n} - b^{2^{n-1}} + 1) " << std::endl;
		ss << "  -f <filename>           input text file (one b per line)" << std::endl;
		ss << "  -d <n> or --device <n>  set device number=<n> (default 0)" << std::endl;
		ss << "  -v or -V                print the startup banner and immediately exit" << std::endl;
#if defined(BOINC)
		ss << "  -boinc                  operate as a BOINC client app" << std::endl;
#endif
		ss << std::endl;
		return ss.str();
	}

public:
	void run(int argc, char * argv[])
	{
		std::vector<std::string> args;
		for (int i = 1; i < argc; ++i) args.push_back(argv[i]);

		bool bBoinc = false;
#if defined(BOINC)
		for (const std::string & arg : args) if (arg == "-boinc") bBoinc = true;
#endif
		pio::getInstance().setBoinc(bBoinc);

		cl_platform_id boinc_platform_id = 0;
		cl_device_id boinc_device_id = 0;
		if (bBoinc)
		{
			BOINC_OPTIONS boinc_options;
			boinc_options_defaults(boinc_options);
			boinc_options.direct_process_action = 0;
			boinc_options.normal_thread_priority = 1;
			const int retval = boinc_init_options(&boinc_options);
			if (retval != 0)
			{
				std::ostringstream ss; ss << "boinc_init returned " << retval;
				throw std::runtime_error(ss.str());
			}
		}

		// if -v or -V then print header and exit
		for (const std::string & arg : args)
		{
			if ((arg[0] == '-') && ((arg[1] == 'v') || (arg[1] == 'V')))
			{
				pio::print(header(args));
				if (bBoinc) boinc_finish(EXIT_SUCCESS);
				return;
			}
		}

		pio::print(header(args, true));

		if (args.empty()) pio::print(usage());	// print usage, display devices and exit

		ocl::platform platform;
		if (platform.displayDevices() == 0) throw std::runtime_error("no OpenCL device");

		size_t d = 0;
		int n = 0;
		std::string filename;
#if defined(BOINC)
		bool ext_device = false;
#endif
		// parse args
		for (size_t i = 0, size = args.size(); i < size; ++i)
		{
			const std::string & arg = args[i];

			if (arg.substr(0, 2) == "-n")
			{
				const std::string nval = ((arg == "-n") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				n = std::atoi(nval.c_str());
				if (n < 8) throw std::runtime_error("n < 8 is not supported");
				if (n > 17) throw std::runtime_error("n > 17 is not supported");
			}
			else if (arg.substr(0, 2) == "-f")
			{
				filename = ((arg == "-f") && (i + 1 < size)) ? args[++i] : arg.substr(2);
			}
			else if (arg.substr(0, 2) == "-d")
			{
				const std::string dev = ((arg == "-d") && (i + 1 < size)) ? args[++i] : arg.substr(2);
				d = size_t(std::atoi(dev.c_str()));
#if defined(BOINC)
				ext_device = true;
#endif
				if (d >= platform.getDeviceCount()) throw std::runtime_error("invalid device number");
			}
			else if (arg.substr(0, 8) == "--device")
			{
				const std::string dev = ((arg == "--device") && (i + 1 < size)) ? args[++i] : arg.substr(8);
				d = size_t(std::atoi(dev.c_str()));
				if (d >= platform.getDeviceCount()) throw std::runtime_error("invalid device number");
#if defined(BOINC)
				ext_device = true;
#endif
			}
		}

		if ((n == 0) || filename.empty())
		{
			/*cyclo & app = cyclo::getInstance();
			engine eng(platform, d, true);
			app.init(8, eng, false);
			app.bench();
			app.release();*/
			return;
		}

#if defined(BOINC)
		if (bBoinc && !boinc_is_standalone() && !ext_device)
		{
			const int err = boinc_get_opencl_ids(argc, argv, 0, &boinc_device_id, &boinc_platform_id);
			if ((err != 0) || (boinc_device_id == 0) || (boinc_platform_id == 0))
			{
				std::ostringstream ss; ss << std::endl << "boinc_get_opencl_ids() failed err = " << err;
				throw std::runtime_error(ss.str());
			}
		}
#endif
		cyclo & app = cyclo::getInstance();
		app.setBoinc(bBoinc);

		const bool is_boinc_platform = bBoinc && (boinc_device_id != 0) && (boinc_platform_id != 0);
		const ocl::platform eng_platform = is_boinc_platform ? ocl::platform(boinc_platform_id, boinc_device_id) : platform;
		const size_t eng_d = is_boinc_platform ? 0 : d;
		engine eng(eng_platform, eng_d, true);

		app.init(n, eng, bBoinc);
		double elapsedTime = 0; const bool success = app.check(filename, elapsedTime);
		app.release();

		uint64_t seconds = uint64_t(elapsedTime), minutes = seconds / 60, hours = minutes / 60;
		seconds -= minutes * 60; minutes -= hours * 60;

		std::stringstream sst;
		sst << "Test is " << (success ? "complete" : "aborted") << ", time = " << std::setfill('0') << std::setw(2)
			<< hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << seconds << "." << std::endl;
		pio::print(sst.str());

		if (bBoinc && success) boinc_finish(BOINC_SUCCESS);
	}
};

int main(int argc, char * argv[])
{
	std::setvbuf(stderr, nullptr, _IONBF, 0);	// no buffer

	try
	{
		application & app = application::getInstance();
		app.run(argc, argv);
	}
	catch (const std::runtime_error & e)
	{
		pio::error(e.what(), true);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
