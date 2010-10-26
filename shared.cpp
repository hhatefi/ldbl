/**

 * File: shared.cpp

 * Author: Dawid Pajak

 */



#ifndef _WIN32
#include <unistd.h>
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <emmintrin.h>
#include <memory.h>
#include "shared.h"
#include "definitions.h"
#include "timer.h"

// note: currently BarrierTree code works correctly only for POWER-OF-TWO number of threads!

BarrierTree::BarrierTree(int threadNum) {
	if (threadNum > 0) {	
		barriers = new BARRIER[threadNum - 1];
	} else {
		barriers = 0;
	}

	count = 0;
	pthread_mutex_init(&lock, 0);
}

BarrierTree::~BarrierTree(void) {
	if (barriers != 0) {
		for (int i = 0; i < count; i++) {
			pthread_barrier_destroy(&barriers[i].barrier);
		}

		delete [] barriers;
	}

	pthread_mutex_destroy(&lock);
}

pthread_barrier_t *BarrierTree::get(int tstart, int tend) {
	//printf("%i %i %i\n", pthread_self(), tstart, tend);
	pthread_mutex_lock(&lock);
	int id = (tend << 16) | tstart;

	// dummy linear search, but should be ok for our purpose (up to 4 cores tests)
	for (int i = 0; i < count; i++) {
		if (id == barriers[i].id) {
			pthread_mutex_unlock(&lock);
			return &barriers[i].barrier;
		}
	}

	BARRIER *barrier = &barriers[count++];
	barrier->id = id;
	pthread_barrier_init(&barrier->barrier, 0, tend - tstart + 1);
	//printf("barrier#%i: tid range: %i-%i\n", count, tstart, tend);
	pthread_mutex_unlock(&lock);
	return &barrier->barrier;
}


void BarrierTree::sync(int tstart, int tend, double *threadWaitTime) {
	double query = 0.0;

	if (threadWaitTime) {
		query = PerfTimer::GetInstance()->peek();
	}

	pthread_barrier_wait(get(tstart, tend));

	if (threadWaitTime) {
		*threadWaitTime += PerfTimer::GetInstance()->peek() - query;
	}
}


void SetupCurrentThread(int cpuId) {
#if !defined(_WIN32) && !defined(_OSX)
	// linux
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(cpuId, &cpuset);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	sched_param sparam;
	sparam.sched_priority = sched_get_priority_max(SCHED_FIFO);
	pthread_setschedparam(pthread_self(), SCHED_FIFO, &sparam);
#else

#ifdef _WIN32
	// win32
	SetThreadAffinityMask(GetCurrentThread(), 1 << cpuId);
#else
	// osx
	sched_param sparam;
	sparam.sched_priority = sched_get_priority_max(SCHED_FIFO);
	pthread_setschedparam(pthread_self(), SCHED_FIFO, &sparam);	
#endif	
#endif	
}

void FlushCacheMemory(void *dst, int size) {
	_mm_mfence();
	char *data = (char *)dst;
	for (int i = 0; i < size; i += CACHE_LINE_SIZE) {
		_mm_clflush(data + i);
	}
}

double GetAbsoluteDiff(float *data1, float *data2, int size) {
	double sum = 0;
	for (int i = 0; i < size; i++) {
		double diff = fabs(data1[i] - data2[i]);
		sum += diff;
	}

	return sum;
}

double GetAbsoluteDiffDP(double *data1, double *data2, int size) {
	double sum = 0;
	for (int i = 0; i < size; i++) {
		double diff = fabs(data1[i] - data2[i]);
		//if (diff > 0) {
		//	printf("%i ", i);
		//}
		sum += diff;
	}

//	printf("\n");
	return sum;
}

static int SortDouble(const void *a, const void *b) {
	if (*((double *)a) < *((double *)b)) {
		return -1;
	}
	if (*((double *)a) > *((double *)b)) {
		return 1;
	}
	return 0;
}

double GetStatTime(double *timing, int size) {
	if (size == 1) {
		return timing[0];
	}

	if (size == 2) {
		return (timing[0] + timing[1]) * 0.5;
	}

	qsort(timing, size, sizeof(double), SortDouble);

	double sum = 0;
	for (int i = 1; i < size - 1; i++) {
		sum += timing[i];
	}

	return sum / (size - 2);
}

void Naive(THREAD_PROC proc, void *data, int outerDim) {
	THREAD_DATA tdata[NUM_THREADS];
	pthread_t threadId[NUM_THREADS];

	int bsize = outerDim / NUM_THREADS;
	if (bsize * NUM_THREADS != outerDim) {
		bsize++;
	}

	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, 0, NUM_THREADS);

	THREAD_DATA *threadData = tdata;
	for (int i = 0; i < NUM_THREADS; i++) {
		//threadData->weights = weights;
		threadData->barrier = &barrier;
		threadData->begin = i * bsize + 1;
		threadData->end = threadData->begin + bsize;
		if (threadData->end > outerDim + 1) {
			threadData->end = outerDim + 1;
		}

		threadData->id = i;
		threadData->data = data;
		pthread_create(&threadId[i], 0, proc, threadData);
		threadData++;
	}

	for(int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threadId[i], 0); 
	}

	pthread_barrier_destroy(&barrier);
}

