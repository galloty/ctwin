/*
Copyright 2023, Yves Gallot

ctwin is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

struct timer
{
	typedef std::chrono::high_resolution_clock::time_point time;

	static time currentTime()
	{
		return std::chrono::high_resolution_clock::now();
	}

	static double diffTime(const time & end, const time & start)
	{
		return std::chrono::duration<double>(end - start).count();
	}

	static std::string formatTime(const double time)
	{
		uint64_t seconds = static_cast<uint64_t>(time), minutes = seconds / 60, hours = minutes / 60;
		seconds -= minutes * 60; minutes -= hours * 60;

		std::stringstream ss;
		ss << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
		return ss.str();
	}
};

class watch
{
private:
	const double _elapsedTime;
	const timer::time _startTime;
	timer::time _currentTime;
	timer::time _recordStartTime;

public:
	watch(const double restoredTime = 0) : _elapsedTime(restoredTime), _startTime(timer::currentTime())
	{
		_currentTime = _recordStartTime = _startTime;
	}

	virtual ~watch() {}

	double getElapsedTime() { _currentTime = timer::currentTime(); return _elapsedTime + timer::diffTime(_currentTime, _startTime); }
	double getRecordTime() { _currentTime = timer::currentTime(); return timer::diffTime(_currentTime, _recordStartTime); }
	void resetRecordTime() { _recordStartTime = _currentTime; }
};
