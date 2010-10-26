#ifndef _SHARED_H
#define _SHARED_H

#ifdef _OSX
#include "pthread_barrier.h"
#endif

#include <pthread.h>
//#include <windows.h>

#define STAT_SAMPLES 1

// fast integer absolute value macro
#define IABS(x) (((x)^((x)>>31))+(((unsigned int)(x))>>31))

#define CLAMP_NEGATIVE(x) ((x)&~((x)>>31))

// core2 L2 cache line size, used by cache flush funtion
#define CACHE_LINE_SIZE 64

#define WEIGHT_TOP    0
#define WEIGHT_LEFT   1
#define WEIGHT_CENTER 2
#define WEIGHT_RIGHT  3
#define WEIGHT_BOTTOM 4
#define WEIGHT_NEAR   5
#define WEIGHT_FAR    6

enum {t_index, x_index, y_index, z_index, DIM};

typedef struct S_THREAD_DATA {
	void *data;
	void *weights;
	int begin, end;
	int timeBlockSize;
	int paddingOffset;
	int spatialDim, temporalDim;
	int id;
	char *flags;
	pthread_barrier_t *barrier;
} THREAD_DATA;

typedef struct S_DOMAIN_1D_CUT {
	int t0, t1;
	int x0, x1;
} DOMAIN_1D_CUT;

typedef struct S_DOMAIN_3D_CUT {
	int t0, t1;
	int x0, x1;
	int y0, y1;
	int z0, z1;
	int dim0[DIM];
	int dim1[DIM];
} DOMAIN_3D_CUT;

typedef struct S_CO_THREAD_DATA {
	int id, depth;
	int tstart, tend;
	double *data;
	void *weights;
	DOMAIN_3D_CUT cut;
	int counter;
	int* lb_array;
	class BarrierTree *barrierTree;
	double threadWaitTime;
} CO_THREAD_DATA;

typedef struct S_BARRIER {
	int id;
	pthread_barrier_t barrier;
} BARRIER;

class BarrierTree {
public:
	BarrierTree(int barrierNum);
	~BarrierTree(void);
	pthread_barrier_t *get(int tstart, int tend);
	void sync(int tstart, int tend, double *threadWaitTime);

private:
	pthread_mutex_t lock;
	BARRIER *barriers;
	int count;
};


typedef void *(*THREAD_PROC)(void *);
typedef void (*FunPtr1)(CO_THREAD_DATA *);
typedef void (*FunPtr2)(void);

void SetupCurrentThread(int cpuId);
void FlushCacheMemory(void *dst, int size);
double GetStatTime(double *timing, int size);
double GetAbsoluteDiff(float *data1, float *data2, int size);

double GetAbsoluteDiffDP(double *data1, double *data2, int size);

void Naive(THREAD_PROC proc, void *data, int outerDim);


#endif

