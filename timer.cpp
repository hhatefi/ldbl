
/**
* File: timer.cpp
* Author: Dawid Pajak
*/

#include "timer.h"

static PerfTimer *sInstance = 0;

PerfTimer *PerfTimer::GetInstance(void) {
	if (!sInstance) {
		sInstance = new PerfTimer();
	}

	return sInstance;
}

#ifdef _WIN32

PerfTimer::PerfTimer() {
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	ifreq = 1.0 / freq.QuadPart;
	QueryPerformanceCounter(&initTime);
}

void PerfTimer::start() {
	QueryPerformanceCounter(&queryStart);
}

double PerfTimer::elapsed() {
	QueryPerformanceCounter(&queryStop);
	double secs = (queryStop.QuadPart - queryStart.QuadPart) * ifreq;
	queryStart = queryStop;

	return secs;
}



double PerfTimer::peek() {
	LARGE_INTEGER query;
	QueryPerformanceCounter(&query);
	return (query.QuadPart - initTime.QuadPart) * ifreq;
}

#else

PerfTimer::PerfTimer() {
	gettimeofday(&initTime, 0);	
}

void PerfTimer::start() {
	gettimeofday(&queryStart, 0);	
}

double PerfTimer::elapsed() {
	gettimeofday(&queryStop, 0);	

	double secs = (double) (queryStop.tv_sec - queryStart.tv_sec + (queryStop.tv_usec - queryStart.tv_usec) * 0.000001);
	queryStart = queryStop;
	return secs;
}

double PerfTimer::peek() {
	timeval query;
	gettimeofday(&query, 0);	
	return (double) (query.tv_sec - initTime.tv_sec + (query.tv_usec - initTime.tv_usec) * 0.000001);
}

#endif

