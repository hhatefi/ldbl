#ifndef _TIMER_H_

#define _TIMER_H_



/**

 * File: timer.h

 * Author: Dawid Pajak

 */



#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#else

#include <sys/time.h>

#endif



class PerfTimer {

public:

	PerfTimer();

	static PerfTimer *GetInstance(void);



	void start();

	double elapsed();

	double peek();

private:



#ifdef _WIN32

	double ifreq;

	LARGE_INTEGER queryStart;

	LARGE_INTEGER queryStop;

	LARGE_INTEGER initTime;

#else

	timeval queryStart;

	timeval queryStop;

	timeval initTime;

#endif

};



#endif



