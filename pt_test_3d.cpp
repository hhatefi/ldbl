
/**
 * File: pt_test_3d.cpp
 * Author: Dawid Pajak
 * 
 * load balancing is not effective as the maximum waiting time is still high and get higher when you increase the number of threads...
 * epsilon is fixed at 1.5
 *  
 */

#include "definitions.h"
#if defined(SIZE_X) && defined(SIZE_Y) && defined(SIZE_Z) && !defined(MATRIX_MODE)


#include <stdio.h>
#include <math.h>
#include <emmintrin.h>
#include <unistd.h>
#include <algorithm>
#include <memory.h>
#include "timer.h"
#include "shared.h"
#include "scheduler.h"
#include <iostream>
using namespace std;
#define X_SHIFT 2

#define X_OFFSET 1
#define Y_OFFSET (SIZE_X+X_SHIFT*2)
#define Z_OFFSET (Y_OFFSET*(SIZE_Y+2))

#define epsilon 1.5

#if _3D
#define DATA_SIZE ((SIZE_X+4)*(SIZE_Y+2))
#elif _2D
#define DATA_SIZE (SIZE_X+4)
#else
#define DATA_SIZE (Z_OFFSET*(SIZE_Z+2))
#endif

double accum = 0;


int MIN_CUT;
int MIN_CUT4D;
int* offset;
int* xdiv_size;
int* ydiv_size;
int* zdiv_size;
int* tdiv_size;
int* lb_array; 
int* lb_array4D;
int* xarr_size;
int* yarr_size;
int* zarr_size;
int* tarr_size;

#if _2D

int M_X_BASE =  16384;
int M_Y_BASE =  1;
int M_Z_BASE =  1;
int M_T_BASE =  8;

int S_X_BASE = 8;
int S_Y_BASE =  1;
int S_Z_BASE =  1;
int S_T_BASE = 8;

#elif _3D

int M_X_BASE =  64;
int M_Y_BASE =  64;
int M_Z_BASE =  1;
int M_T_BASE =  128;

int S_X_BASE =  64;
int S_Y_BASE =  8;
int S_Z_BASE =  1;
int S_T_BASE =  128;

#else

int M_X_BASE =  256;
int M_Y_BASE =  8;
int M_Z_BASE =  8;
int M_T_BASE =  128;


int S_X_BASE =  256;
int S_Y_BASE =  8;
int S_Z_BASE =  8;
int S_T_BASE =  128;

#endif



int M_BASE[DIM];
int S_BASE[DIM];

enum { SCHEDULER_PARALLELISM, SCHEDULER_FORM, SCHEDULER_NUM };
enum { CURRENT_SCHEDULER= SCHEDULER_PARALLELISM };
	
static void PCOTraverser(CO_THREAD_DATA *threadData);

static void *ATS1DP(void *ptr) {
	THREAD_DATA *threadData = (THREAD_DATA *)ptr;
	SetupCurrentThread(threadData->id);

	double *data = (double *)threadData->data;
	int tbsize = threadData->timeBlockSize;
	int begin = threadData->begin;
	int end = threadData->end;
	int point = end - (tbsize << 1) - 1;
	int tend = (int)ceil((double)TIME_STEPS / tbsize);

	for (int tb = 0; tb < tend; tb++) {
		for (int z = begin; z < end; z++) {
			int tst = tb * tbsize;
			int tmp = ((z - point) >> 1) + ((z - point) & 1);
			tst += CLAMP_NEGATIVE(tmp);

			int ten = tb * tbsize + tbsize - 1; 
			tmp = (z - begin + (tb * tbsize << 1)) >> 1;
			if (tmp < ten) {
				ten = tmp;
			}

			int t1 = tst % tbsize;
			for (int t = tst; t <= ten; t++, t1++) {
				if (t1 == tbsize) {
					t1 -= tbsize;
				}

				if (z - t1 < 1 || z - t1 > SIZE_Z) {
					continue;
				}

				if (t > TIME_STEPS - 1) {
					break;
				}
			
				__m128d *p = (__m128d *)(data + (t & 1) * DATA_SIZE + (z - t1) * Z_OFFSET + 1 * Y_OFFSET + X_SHIFT);
				__m128d *q = (__m128d *)(data + ((t + 1) & 1) * DATA_SIZE + (z - t1) * Z_OFFSET + 1 * Y_OFFSET + X_SHIFT);

				for (int y = 1; y <= SIZE_Y; y++) {
					__m128d currVec = p[0];
					__m128d leftVec = _mm_shuffle_pd(_mm_setzero_pd(), currVec, 0x01);
					for (int x = 0; x < (SIZE_X >> 1); x++, p++, q++) {
						__m128d nextVec = p[1];
						__m128d result = _mm_mul_pd(currVec, _mm_set1_pd(0.4));
						result = _mm_add_pd(result, _mm_mul_pd(leftVec, _mm_set1_pd(0.1)));
						__m128d rightVec = _mm_shuffle_pd(currVec, nextVec, 0x01);
						result = _mm_add_pd(result, _mm_mul_pd(rightVec, _mm_set1_pd(0.3)));
						result = _mm_add_pd(result, _mm_mul_pd(p[-Y_OFFSET >> 1], _mm_set1_pd(0.03)));
						result = _mm_add_pd(result, _mm_mul_pd(p[Y_OFFSET >> 1], _mm_set1_pd(0.07)));
						result = _mm_add_pd(result, _mm_mul_pd(p[-Z_OFFSET >> 1], _mm_set1_pd(0.03)));
						result = _mm_add_pd(result, _mm_mul_pd(p[Z_OFFSET >> 1], _mm_set1_pd(0.07)));
						q[0] = result;

						currVec = nextVec;
						leftVec = rightVec;
					}

					// skip padding
					p += X_SHIFT;
					q += X_SHIFT;
				}
			}
		}

		pthread_barrier_wait(threadData->barrier);
	}

	return 0;
}

static void *ATS2DP(void *ptr){
	THREAD_DATA *threadData = (THREAD_DATA *)ptr;
	SetupCurrentThread(threadData->id);

	double *data = (double *)threadData->data;
	int tbsize = threadData->timeBlockSize;
	int offset = (tbsize >> 1) - 1;

	volatile char *flags = threadData->flags;
	int sd = threadData->paddingOffset;
   	int id = threadData->id;
	int spatialDim = threadData->spatialDim;
	int temporalDim = threadData->temporalDim;
   	int totalDiamonds = ((spatialDim << 1) + 1) * (temporalDim >> 1) + (temporalDim & 1) * spatialDim;

  	int tsize = tbsize - 1;	 
	//int cntr = -(tsize >> 1);

	for (int i = id; i < totalDiamonds; i += NUM_THREADS){		
		int tmpx = i / ((spatialDim << 1) + 1);
		int tmpy = i - tmpx * ((spatialDim << 1) + 1);
		
		int ycoord, xcoord;
		if (tmpy > (spatialDim - 1)) {
			ycoord = (tmpx << 1) + 1;
			xcoord = tmpy - spatialDim;
		} else {
			ycoord = tmpx << 1;
			xcoord = tmpy;
		}
		
		int tb  = ((ycoord * tbsize) >> 1) - (tsize >> 1);			
		int begin = (ycoord & 1) * (-tbsize >> 1) + xcoord * tbsize - sd;
	
		int x1 = xcoord;
		if (ycoord & 1) {
			x1--;
		} else {
			x1++;
		}	
		x1 = CLAMP_NEGATIVE(x1);
		
		int x2 = xcoord;
		if (x2 == spatialDim) {
			x2--;
		}

		if (ycoord != 0) {
			int index = (ycoord - 1) * (spatialDim + 1);
			while (flags[index + x1] == 0 || flags[index + x2] == 0) {
				sched_yield();
			}
		}

		for (int z = 1; z < (SIZE_Z + tbsize); z++) {
			int tst = z - SIZE_Z;
			if (tst < 0) {
				tst = 0;
			}

			if (tst + tb < 0) {
				tst -= tb + tst;
			}

			int ten = z;
			if (ten + tb > TIME_STEPS) {
				ten = TIME_STEPS - tb;
			}

			for (int t = tst; t < ten; t++) {			
				int tmp = IABS(t - offset);
				int yst =  begin + tmp;
				if (yst < 1) {
					yst = 1;
				}

				int yen =  begin + tbsize - tmp;
				if (yen > SIZE_Y + 1) {
					yen = SIZE_Y + 1;
				}
		
				int t1 = t + tb;
				__m128d *p = (__m128d *)(data + (t1 & 1) * DATA_SIZE + (z - t) * Z_OFFSET + yst * Y_OFFSET + X_SHIFT);
				__m128d *q = (__m128d *)(data + ((t1 + 1) & 1) * DATA_SIZE + (z - t) * Z_OFFSET + yst * Y_OFFSET + X_SHIFT);

				for (int y = yst; y < yen; y++) {
					__m128d currVec = p[0];
					__m128d leftVec = _mm_shuffle_pd(_mm_setzero_pd(), currVec, 0x01);
					for (int x = 0; x < (SIZE_X >> 1); x++, p++, q++) {
						__m128d nextVec = p[1];
						__m128d result = _mm_mul_pd(currVec, _mm_set1_pd(0.4));
						result = _mm_add_pd(result, _mm_mul_pd(leftVec, _mm_set1_pd(0.1)));
						__m128d rightVec = _mm_shuffle_pd(currVec, nextVec, 0x01);
						result = _mm_add_pd(result, _mm_mul_pd(rightVec, _mm_set1_pd(0.3)));
						result = _mm_add_pd(result, _mm_mul_pd(p[-Y_OFFSET >> 1], _mm_set1_pd(0.03)));
						result = _mm_add_pd(result, _mm_mul_pd(p[Y_OFFSET >> 1], _mm_set1_pd(0.07)));
						result = _mm_add_pd(result, _mm_mul_pd(p[-Z_OFFSET >> 1], _mm_set1_pd(0.03)));
						result = _mm_add_pd(result, _mm_mul_pd(p[Z_OFFSET >> 1], _mm_set1_pd(0.07)));
						q[0] = result;

						currVec = nextVec;
						leftVec = rightVec;
					}

					// skip padding
					p += X_SHIFT;
					q += X_SHIFT;
				}
			}
		}

		flags[ycoord * (spatialDim + 1) + xcoord] = 1;
	}	

	return 0;
}
#if _2D

static void *NaiveDP(void *ptr) {
	

	THREAD_DATA *threadData = (THREAD_DATA *)ptr;
	SetupCurrentThread(threadData->id);

	double *data = (double *)threadData->data;

	for (int t = 0; t < TIME_STEPS; t++) {
		
		__m128d *p = (__m128d *)(data + (t & 1) * DATA_SIZE + X_SHIFT);
		__m128d *q = (__m128d *)(data + ((t + 1) & 1) * DATA_SIZE + X_SHIFT);

		__m128d currVec = p[0];
		__m128d leftVec = _mm_shuffle_pd(_mm_setzero_pd(), currVec, 0x01);
		for (int x = 0; x < (SIZE_X >> 1); x++, p++, q++) {
			__m128d nextVec = p[1];
			__m128d result = _mm_mul_pd(currVec, _mm_set1_pd(0.4));
			result = _mm_add_pd(result, _mm_mul_pd(leftVec, _mm_set1_pd(0.1)));
			__m128d rightVec = _mm_shuffle_pd(currVec, nextVec, 0x01);
			result = _mm_add_pd(result, _mm_mul_pd(rightVec, _mm_set1_pd(0.3)));
				
			q[0] = result;

			currVec = nextVec;
			leftVec = rightVec;
		}

		
		
		pthread_barrier_wait(threadData->barrier);
	}
	
	return 0;
}
#elif _3D

static void *NaiveDP(void *ptr) {
	

	THREAD_DATA *threadData = (THREAD_DATA *)ptr;
	SetupCurrentThread(threadData->id);

	double *data = (double *)threadData->data;

	for (int t = 0; t < TIME_STEPS; t++) {
		for (int y = threadData->begin; y < threadData->end; y++) {
			__m128d *p = (__m128d *)(data + (t & 1) * DATA_SIZE + y * Y_OFFSET + X_SHIFT);
			__m128d *q = (__m128d *)(data + ((t + 1) & 1) * DATA_SIZE + y * Y_OFFSET + X_SHIFT);

			__m128d currVec = p[0];
			__m128d leftVec = _mm_shuffle_pd(_mm_setzero_pd(), currVec, 0x01);
			for (int x = 0; x < (SIZE_X >> 1); x++, p++, q++) {
				__m128d nextVec = p[1];
				__m128d result = _mm_mul_pd(currVec, _mm_set1_pd(0.4));
				result = _mm_add_pd(result, _mm_mul_pd(leftVec, _mm_set1_pd(0.1)));
				__m128d rightVec = _mm_shuffle_pd(currVec, nextVec, 0x01);
				result = _mm_add_pd(result, _mm_mul_pd(rightVec, _mm_set1_pd(0.3)));
				result = _mm_add_pd(result, _mm_mul_pd(p[-Y_OFFSET >> 1], _mm_set1_pd(0.03)));
				result = _mm_add_pd(result, _mm_mul_pd(p[Y_OFFSET >> 1], _mm_set1_pd(0.07)));
				
				q[0] = result;

				currVec = nextVec;
				leftVec = rightVec;
			}

			// skip padding
			p += X_SHIFT;
			q += X_SHIFT;
		}
		pthread_barrier_wait(threadData->barrier);
	}

	return 0;
}

#else
static void *NaiveDP(void *ptr) {
	THREAD_DATA *threadData = (THREAD_DATA *)ptr;
	SetupCurrentThread(threadData->id);

	double *data = (double *)threadData->data;

	for (int t = 0; t < TIME_STEPS; t++) {
		for (int z = threadData->begin; z < threadData->end; z++) {
			__m128d *p = (__m128d *)(data + (t & 1) * DATA_SIZE + z * Z_OFFSET + 1 * Y_OFFSET + X_SHIFT);
			__m128d *q = (__m128d *)(data + ((t + 1) & 1) * DATA_SIZE + z * Z_OFFSET + 1 * Y_OFFSET + X_SHIFT);

			// naive with manual vectorization
			for (int y = 1; y <= SIZE_Y; y++) {
				__m128d currVec = p[0];
				__m128d leftVec = _mm_shuffle_pd(_mm_setzero_pd(), currVec, 0x01);
				for (int x = 0; x < (SIZE_X >> 1); x++, p++, q++) {
					__m128d nextVec = p[1];
					__m128d result = _mm_mul_pd(currVec, _mm_set1_pd(0.4));
					result = _mm_add_pd(result, _mm_mul_pd(leftVec, _mm_set1_pd(0.1)));
					__m128d rightVec = _mm_shuffle_pd(currVec, nextVec, 0x01);
					result = _mm_add_pd(result, _mm_mul_pd(rightVec, _mm_set1_pd(0.3)));
					result = _mm_add_pd(result, _mm_mul_pd(p[-Y_OFFSET >> 1], _mm_set1_pd(0.03)));
					result = _mm_add_pd(result, _mm_mul_pd(p[Y_OFFSET >> 1], _mm_set1_pd(0.07)));
					result = _mm_add_pd(result, _mm_mul_pd(p[-Z_OFFSET >> 1], _mm_set1_pd(0.03)));
					result = _mm_add_pd(result, _mm_mul_pd(p[Z_OFFSET >> 1], _mm_set1_pd(0.07)));
					q[0] = result;

					currVec = nextVec;
					leftVec = rightVec;
				}

				// skip padding
				p += X_SHIFT;
				q += X_SHIFT;
			}
		}

		pthread_barrier_wait(threadData->barrier);
	}

	return 0;
}
#endif

#if _3D
static void InitDataDP(double *data) {
	double val = 1.0;
	for (int j = 0; j < SIZE_Y + 2; j++) {
		for (int k = 0; k < (SIZE_X + 2 * X_SHIFT); k++) {
			int index = j * Y_OFFSET + k * X_OFFSET;
			if (j == 0 || j == SIZE_Y + 1 || k < X_SHIFT || k >= SIZE_X + X_SHIFT) {
				data[index] = 0.0;
			} else {
				data[index] = val;
				val++;
			}
		}
	}
	
}
#elif _2D
static void InitDataDP(double *data) {
	double val = 1.0;
	for (int k = 0; k < (SIZE_X + 2 * X_SHIFT); k++) {
		int index = k * X_OFFSET;
		if (k < X_SHIFT || k >= SIZE_X + X_SHIFT) {
			data[index] = 0.0;
		} else {
			data[index] = val;
			val++;
		}
	}
}
#else
static void InitDataDP(double *data) {
	double val = 1.0;
	for (int i = 0; i < SIZE_Z + 2; i++) {
		for (int j = 0; j < SIZE_Y + 2; j++) {
			for (int k = 0; k < (SIZE_X + 2 * X_SHIFT); k++) {
				int index = i * Z_OFFSET + j * Y_OFFSET + k * X_OFFSET;
				if (i == 0 || j == 0 || i == SIZE_Z + 1 || j == SIZE_Y + 1 || k < X_SHIFT || k >= SIZE_X + X_SHIFT) {
					data[index] = 0.0;
				} else {
					data[index] = val;
					val++;
				}
			}
		}
	}
}
#endif
/*
inline void COComputeKernel(CO_THREAD_DATA *tdata){
	
	DOMAIN_3D_CUT *cut = &tdata->cut;
	double *data = (double *)tdata->data;
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	
	int z0 = cut->dim0[z_index];
	int z1 = cut->dim1[z_index];
	

	
	for(int t = t0; t < t1; t++) {
	
		for(int i = (z0-t); i < (z1-t); i++){
		for(int j = (y0-t); j < (y1-t); j++){
		for(int k = (x0-t); k < (x1-t); k++){
						
			if(t < TIME_STEPS && i <= SIZE_Z && i > 0 && j > 0 && j <= SIZE_Y && k < SIZE_X && k >= 0){
				//cout<<"here"<<endl;		
				double sum = 0.0;
				int index = i * Z_OFFSET + j * Y_OFFSET + k * X_OFFSET;
				
				sum = 0.4 * data[(t & 1) * DATA_SIZE + index];
				
				int index1 = i * Z_OFFSET + j * Y_OFFSET + (k-1) * X_OFFSET;
				
				if(k != 0){
					sum += 0.1 * data[(t & 1) * DATA_SIZE + index1]; 
				}
				
				int index2 = i * Z_OFFSET + j * Y_OFFSET + (k+1) * X_OFFSET;
				
				if(k != SIZE_X-1){
					sum += 0.3 * data[(t & 1) * DATA_SIZE + index2]; 
				}
				
				int index3 = i * Z_OFFSET + (j-1) * Y_OFFSET + k * X_OFFSET;
				sum += 0.03 * data[(t & 1) * DATA_SIZE + index3];
				
				int index4 = i * Z_OFFSET + (j+1) * Y_OFFSET + k * X_OFFSET;
				sum += 0.07 * data[(t & 1) * DATA_SIZE + index4];
				
				int index5 = (i-1) * Z_OFFSET + j * Y_OFFSET + k * X_OFFSET;
				sum += 0.03 * data[(t & 1) * DATA_SIZE + index5];
				
				int index6 = (i+1) * Z_OFFSET + j * Y_OFFSET + k * X_OFFSET;
				sum += 0.07 * data[(t & 1) * DATA_SIZE + index6];
				
				data[((t+1) & 1) * DATA_SIZE + index] = sum;
				//cout<<data[((t+1) & 1) * DATA_SIZE + index]<<endl;
					
			}
		}}}
	}
			
}
/**/
#if _3D
inline void COComputeKernel(CO_THREAD_DATA *tdata){
	
	
	DOMAIN_3D_CUT *cut = &tdata->cut;
	double *data = (double *)tdata->data;
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	//cout<<t0<<" "<<t1<<endl;
	
	if (t1 > TIME_STEPS) {
		t1 = TIME_STEPS;
	}
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	//cout<<x0<<" "<<x1<<endl;
	
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	//cout<<y0<<" "<<y1<<endl;
	
	
	
	//cout<<z0<<" "<<z1<<endl;
	
	//char tt;
	//cin>>tt;
	
	for (int t = t0; t < t1; t++) {
		int xst = max(0, x0 - t);
		int xen = min(SIZE_X, x1 - t);
		

		if (xst < xen) {
			int yst = max(0, y0 - t);
			int yen = min(SIZE_Y, y1 - t);
			

			double *ps = tdata->data + (t & 1) * DATA_SIZE + (yst + 1) * Y_OFFSET + X_SHIFT;
			double *qs = tdata->data + ((t + 1) & 1) * DATA_SIZE + (yst + 1) * Y_OFFSET + X_SHIFT;

			
			for (int j = yst; j < yen; j++) {
					
					
				// process starting unaligned element
				int idx = xst;
				if ((idx & 0x1) != 0) {
					double result = 0.4 * ps[idx];
					result += 0.1 * ps[idx - 1];
					result += 0.3 * ps[idx + 1];
					result += 0.03 * ps[idx - Y_OFFSET];
					result += 0.07 * ps[idx + Y_OFFSET];
					
					qs[idx] = result;
					idx++;
				}

				__m128d *p = (__m128d *)(ps + idx);
				__m128d *q = (__m128d *)(qs + idx);
				__m128d currVec = p[0];
				__m128d leftVec = _mm_shuffle_pd(p[-1], currVec, 0x01);

				int vlength = (xen - idx) >> 1;
				idx += vlength << 1;

				while (vlength > 0) {
					__m128d nextVec = p[1];
					__m128d result = _mm_mul_pd(currVec, _mm_set1_pd(0.4));
					result = _mm_add_pd(result, _mm_mul_pd(leftVec, _mm_set1_pd(0.1)));
					__m128d rightVec = _mm_shuffle_pd(currVec, nextVec, 0x01);
					result = _mm_add_pd(result, _mm_mul_pd(rightVec, _mm_set1_pd(0.3)));
					result = _mm_add_pd(result, _mm_mul_pd(p[-Y_OFFSET >> 1], _mm_set1_pd(0.03)));
					result = _mm_add_pd(result, _mm_mul_pd(p[Y_OFFSET >> 1], _mm_set1_pd(0.07)));
					
					q[0] = result;

					currVec = nextVec;
					leftVec = rightVec;

					p++;
					q++;
					vlength--;
				}

				// process unaligned end element
				if ((xen & 1) != 0 && idx < xen) {
					double result = 0.4 * ps[idx];
					result += 0.1 * ps[idx - 1];
					result += 0.3 * ps[idx + 1];
					result += 0.03 * ps[idx - Y_OFFSET];
					result += 0.07 * ps[idx + Y_OFFSET];
					
					qs[idx] = result;
				}

				ps += Y_OFFSET;
				qs += Y_OFFSET;
			}
		}
	}
}

#elif _2D
inline void COComputeKernel(CO_THREAD_DATA *tdata){
	
	
	DOMAIN_3D_CUT *cut = &tdata->cut;
	
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	//cout<<t0<<" "<<t1<<endl;
	
	if (t1 > TIME_STEPS) {
		t1 = TIME_STEPS;
	}
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	//cout<<x0<<" "<<x1<<endl;
	
	
	
	for (int t = t0; t < t1; t++) {
		int xst = max(0, x0 - t);
		int xen = min(SIZE_X, x1 - t);
		
		if (xst < xen) {
			
			double *ps = tdata->data + (t & 1) * DATA_SIZE + X_SHIFT;
			double *qs = tdata->data + ((t + 1) & 1) * DATA_SIZE + X_SHIFT;

			
					
					
					// process starting unaligned element
					int idx = xst;
					if ((idx & 0x1) != 0) {
						double result = 0.4 * ps[idx];
						result += 0.1 * ps[idx - 1];
						result += 0.3 * ps[idx + 1];
						
						qs[idx] = result;
						idx++;
					}

					__m128d *p = (__m128d *)(ps + idx);
					__m128d *q = (__m128d *)(qs + idx);
					__m128d currVec = p[0];
					__m128d leftVec = _mm_shuffle_pd(p[-1], currVec, 0x01);

					int vlength = (xen - idx) >> 1;
					idx += vlength << 1;

					while (vlength > 0) {
						__m128d nextVec = p[1];
						__m128d result = _mm_mul_pd(currVec, _mm_set1_pd(0.4));
						result = _mm_add_pd(result, _mm_mul_pd(leftVec, _mm_set1_pd(0.1)));
						__m128d rightVec = _mm_shuffle_pd(currVec, nextVec, 0x01);
						result = _mm_add_pd(result, _mm_mul_pd(rightVec, _mm_set1_pd(0.3)));
						
						q[0] = result;

						currVec = nextVec;
						leftVec = rightVec;

						p++;
						q++;
						vlength--;
					}

					// process unaligned end element
					if ((xen & 1) != 0 && idx < xen) {
						double result = 0.4 * ps[idx];
						result += 0.1 * ps[idx - 1];
						result += 0.3 * ps[idx + 1];
						
						qs[idx] = result;
					}

					//ps += X_SHIFT;
					//qs += X_SHIFT;
				}
			}
}
#else
inline void COComputeKernel(CO_THREAD_DATA *tdata){
	
	//cout<<"entered here"<<endl;
	DOMAIN_3D_CUT *cut = &tdata->cut;
	double *data = (double *)tdata->data;
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	//cout<<t0<<" "<<t1<<endl;
	
	if (t1 > TIME_STEPS) {
		t1 = TIME_STEPS;
	}
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	//cout<<x0<<" "<<x1<<endl;
	
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	//cout<<y0<<" "<<y1<<endl;
	
	
	int z0 = cut->dim0[z_index];
	int z1 = cut->dim1[z_index];
	//cout<<z0<<" "<<z1<<endl;
	
	//char tt;
	//cin>>tt;
	
	for (int t = t0; t < t1; t++) {
		int xst = max(0, x0 - t);
		int xen = min(SIZE_X, x1 - t);
		/*if (xst < 0) {
			xst = 0;
		}
		if (xen > SIZE_X) {
			xen = SIZE_X;
		}*/

		if (xst < xen) {
			int yst = max(0, y0 - t);
			int yen = min(SIZE_Y, y1 - t);
			/*if (yst < 0) {
				yst = 0;
			}
			if (yen > SIZE_Y) {
				yen = SIZE_Y;
			}*/

			int zst = max(0, z0 - t);
			int zen = min(SIZE_Z, z1 - t);
			/*if (zst < 0) {
				zst = 0;
			}
			if (zen > SIZE_Z) {
				zen = SIZE_Z;
			}*/

			double *ps = tdata->data + (t & 1) * DATA_SIZE + (zst + 1) * Z_OFFSET + (yst + 1) * Y_OFFSET + X_SHIFT;
			double *qs = tdata->data + ((t + 1) & 1) * DATA_SIZE + (zst + 1) * Z_OFFSET + (yst + 1) * Y_OFFSET + X_SHIFT;

			for (int i = zst; i < zen; i++) {
				for (int j = yst; j < yen; j++) {
					
					
					// process starting unaligned element
					int idx = xst;
					if ((idx & 0x1) != 0) {
						double result = 0.4 * ps[idx];
						result += 0.1 * ps[idx - 1];
						result += 0.3 * ps[idx + 1];
						result += 0.03 * ps[idx - Y_OFFSET];
						result += 0.07 * ps[idx + Y_OFFSET];
						result += 0.03 * ps[idx - Z_OFFSET];
						result += 0.07 * ps[idx + Z_OFFSET];
						qs[idx] = result;
						idx++;
					}

					__m128d *p = (__m128d *)(ps + idx);
					__m128d *q = (__m128d *)(qs + idx);
					__m128d currVec = p[0];
					__m128d leftVec = _mm_shuffle_pd(p[-1], currVec, 0x01);

					int vlength = (xen - idx) >> 1;
					idx += vlength << 1;

					while (vlength > 0) {
						__m128d nextVec = p[1];
						__m128d result = _mm_mul_pd(currVec, _mm_set1_pd(0.4));
						result = _mm_add_pd(result, _mm_mul_pd(leftVec, _mm_set1_pd(0.1)));
						__m128d rightVec = _mm_shuffle_pd(currVec, nextVec, 0x01);
						result = _mm_add_pd(result, _mm_mul_pd(rightVec, _mm_set1_pd(0.3)));
						result = _mm_add_pd(result, _mm_mul_pd(p[-Y_OFFSET >> 1], _mm_set1_pd(0.03)));
						result = _mm_add_pd(result, _mm_mul_pd(p[Y_OFFSET >> 1], _mm_set1_pd(0.07)));
						result = _mm_add_pd(result, _mm_mul_pd(p[-Z_OFFSET >> 1], _mm_set1_pd(0.03)));
						result = _mm_add_pd(result, _mm_mul_pd(p[Z_OFFSET >> 1], _mm_set1_pd(0.07)));
						q[0] = result;

						currVec = nextVec;
						leftVec = rightVec;

						p++;
						q++;
						vlength--;
					}

					// process unaligned end element
					if ((xen & 1) != 0 && idx < xen) {
						double result = 0.4 * ps[idx];
						result += 0.1 * ps[idx - 1];
						result += 0.3 * ps[idx + 1];
						result += 0.03 * ps[idx - Y_OFFSET];
						result += 0.07 * ps[idx + Y_OFFSET];
						result += 0.03 * ps[idx - Z_OFFSET];
						result += 0.07 * ps[idx + Z_OFFSET];
						qs[idx] = result;
					}

					ps += Y_OFFSET;
					qs += Y_OFFSET;
				}

				int stride = Z_OFFSET - Y_OFFSET * (yen - yst);
				ps += stride;
				qs += stride;
			}
		}
	}
}
#endif

#if 1
void* COTraverser(CO_THREAD_DATA *tdata) {
	
	int dim0[4];
	int dim1[4];
	int wdim[4];
	
	int cind = 0;
	
	DOMAIN_3D_CUT *cut = &tdata->cut;
	
	for(int i = 0; i < DIM; i++){
		dim0[i] = cut->dim0[i];
		dim1[i] = cut->dim1[i];
		wdim[i] = cut->dim1[i] - cut->dim0[i];
	}
	
	int relSize[]= {
   		wdim[0]/S_BASE[0],
   		wdim[1]/S_BASE[1],
	   	wdim[2]/S_BASE[2],
	   	wdim[3]/S_BASE[3],
	};
	
	int maxSize = 1;
	 
	for(int i = 0; i < DIM; i++) {
		if(maxSize<relSize[i]){ 
			maxSize = relSize[i];
			cind = i;
		}
	}
	if(maxSize == 1){
		/*for(int i = 0; i < DIM; i++){
			if(wdim[i] > S_BASE[i]){
				cout<<"stop!"<<endl;
				char tt;
				cin>>tt;
			}
		}*/
		COComputeKernel(tdata);
		return 0;
	}
	else{
		
		//cout<<"here"<<endl;
		for(int i = 0; i < DIM; i++){
			if(i == cind){
				cut->dim0[i] = dim0[i];
				cut->dim1[i] = dim0[i] + (wdim[i] >> 1);
			}
			else{
				cut->dim0[i] = dim0[i];
				cut->dim1[i] = dim1[i];
			}
		}
		
		
		
		
		tdata->depth++;
		COTraverser(tdata);
		tdata->depth--;
		
		for(int i = 0; i < DIM; i++){
			if(i == cind){
				cut->dim0[i] = dim0[i] + (wdim[i] >> 1);
				cut->dim1[i] = dim1[i];
			}
			else{
				cut->dim0[i] = dim0[i];
				cut->dim1[i] = dim1[i];
			}
		}
		
		
		
		tdata->depth++;
		COTraverser(tdata);
		tdata->depth--;
			
	}
	/*
	if (wdim[y_index] == maxSize) {
		
		cut->dim0[t_index] = dim0[t_index];
		cut->dim1[t_index] = dim1[t_index];
		cut->dim0[y_index] = dim0[y_index];
		cut->dim1[y_index] = dim0[y_index] + (wdim[y_index] >> 1);
		cut->dim0[x_index] = dim0[x_index];
		cut->dim1[x_index] = dim1[x_index];
		cut->dim0[z_index] = dim0[z_index];
		cut->dim1[z_index] = dim1[z_index];
		
		tdata->depth++;
		COTraverser(tdata);
		tdata->depth--;
		
		cut->dim0[t_index] = dim0[t_index];
		cut->dim1[t_index] = dim1[t_index];	
		cut->dim0[y_index] = dim0[y_index] + (wdim[y_index] >> 1);
		cut->dim1[y_index] = dim1[y_index];
		cut->dim0[x_index] = dim0[x_index];
		cut->dim1[x_index] = dim1[x_index];
		cut->dim0[z_index] = dim0[z_index];
		cut->dim1[z_index] = dim1[z_index];
		
		tdata->depth++;			
		COTraverser(tdata);
		tdata->depth--;
		
		return 0;
				
	}
	else if (wdim[z_index] == maxSize) {
		
		cut->dim0[t_index] = dim0[t_index];
		cut->dim1[t_index] = dim1[t_index];
		cut->dim0[z_index] = dim0[z_index];
		cut->dim1[z_index] = dim0[z_index] + (wdim[z_index] >> 1);
		cut->dim0[x_index] = dim0[x_index];
		cut->dim1[x_index] = dim1[x_index];
		cut->dim0[y_index] = dim0[y_index];
		cut->dim1[y_index] = dim1[y_index];
		
		tdata->depth++;			
		COTraverser(tdata);
		tdata->depth--;
		
		
		cut->dim0[t_index] = dim0[t_index];
		cut->dim1[t_index] = dim1[t_index];
		cut->dim0[z_index] = dim0[z_index] + (wdim[z_index] >> 1);
		cut->dim1[z_index] = dim1[z_index];
		cut->dim0[x_index] = dim0[x_index];
		cut->dim1[x_index] = dim1[x_index];
		cut->dim0[y_index] = dim0[y_index];
		cut->dim1[y_index] = dim1[y_index];
		
		tdata->depth++;			
		COTraverser(tdata);
		tdata->depth--;
		
		
		return 0;		
	}
	else if (wdim[t_index] == maxSize) {
		
		cut->dim0[t_index] = dim0[t_index];
		cut->dim1[t_index] = dim0[t_index] + (wdim[t_index] >> 1);
		cut->dim0[x_index] = dim0[x_index];
		cut->dim1[x_index] = dim1[x_index];
		cut->dim0[y_index] = dim0[y_index];
		cut->dim1[y_index] = dim1[y_index];
		cut->dim0[z_index] = dim0[z_index];
		cut->dim1[z_index] = dim1[z_index];
		
		tdata->depth++;			
		COTraverser(tdata);
		tdata->depth--;
		
		
		cut->dim0[t_index] = dim0[t_index] + (wdim[t_index] >> 1);
		cut->dim1[t_index] = dim1[t_index];
		cut->dim0[x_index] = dim0[x_index];
		cut->dim1[x_index] = dim1[x_index];
		cut->dim0[y_index] = dim0[y_index];
		cut->dim1[y_index] = dim1[y_index];
		cut->dim0[z_index] = dim0[z_index];
		cut->dim1[z_index] = dim1[z_index];
			
		
		tdata->depth++;			
		COTraverser(tdata);
		tdata->depth--;
		
		return 0;
			
	}
	else if (wdim[x_index] == maxSize) {
		
		cut->dim0[t_index] = dim0[t_index];
		cut->dim1[t_index] = dim1[t_index];
		cut->dim0[x_index] = dim0[x_index];
		cut->dim1[x_index] = dim0[x_index] + (wdim[x_index] >> 1);
		cut->dim0[y_index] = dim0[y_index];
		cut->dim1[y_index] = dim1[y_index];
		cut->dim0[z_index] = dim0[z_index];
		cut->dim1[z_index] = dim1[z_index];
		
		tdata->depth++;			
		COTraverser(tdata);
		tdata->depth--;
		
		
		cut->dim0[t_index] = dim0[t_index];
		cut->dim1[t_index] = dim1[t_index];
		cut->dim0[x_index] = dim0[x_index] + wdim[x_index] >> 1;
		cut->dim1[x_index] = dim1[x_index];
		cut->dim0[y_index] = dim0[y_index];
		cut->dim1[y_index] = dim1[y_index];
		cut->dim0[z_index] = dim0[z_index];
		cut->dim1[z_index] = dim1[z_index];
			
		
		tdata->depth++;			
		COTraverser(tdata);
		tdata->depth--;
		
		return 0;
			
	}*/
}
#else
static void COTraverser(CO_THREAD_DATA *tdata) {
	DOMAIN_3D_CUT *cut = &tdata->cut;
	
	int t0 = cut->dim0[t_index];
	int x0 = cut->dim0[x_index];
	int y0 = cut->dim0[y_index];
	int z0 = cut->dim0[z_index];
	int wt = cut->dim1[t_index] - t0;
	int wx = cut->dim1[x_index] - x0;
	int wy = cut->dim1[y_index] - y0;
	int wz = cut->dim1[z_index] - z0;
	
	int maxd = max(wx, max(max(wy, wz), wt));
	
	if (wx == maxd && wx > S_X_BASE) {
		
		cut->dim0[t_index] = t0;
		cut->dim1[t_index] = t0 + wt;
		
		cut->dim0[x_index] = x0;
		cut->dim1[x_index] = x0 + (wx >> 1);
		
		cut->dim0[y_index] = y0;
		cut->dim1[y_index] = y0 + wy;
		
		cut->dim0[z_index] = z0;
		cut->dim1[z_index] = z0 + wz;
		
		COTraverser(tdata);
		
		cut->dim0[t_index] = t0;
		cut->dim1[t_index] = t0 + wt;
		
		cut->dim0[x_index] = x0 + (wx >> 1);
		cut->dim1[x_index] = x0 + wx;
		
		
		cut->dim0[y_index] = y0;
		cut->dim1[y_index] = y0 + wy;
		
		cut->dim0[z_index] = z0;
		cut->dim1[z_index] = z0 + wz;
		
			
		COTraverser(tdata);
		
		return;
		
			
	} 
	else if (wy == maxd && wy > S_Y_BASE) {
		
		cut->dim0[t_index] = t0;
		cut->dim1[t_index] = t0 + wt;
		
		cut->dim0[y_index] = y0;
		cut->dim1[y_index] = y0 + (wy >> 1);
		
		cut->dim0[x_index] = x0;
		cut->dim1[x_index] = x0 + wx;
		
		cut->dim0[z_index] = z0;
		cut->dim1[z_index] = z0 + wz;
		
		COTraverser(tdata);
		
		cut->dim0[t_index] = t0;
		cut->dim1[t_index] = t0 + wt;	
		
		cut->dim0[y_index] = y0 + (wy >> 1);
		cut->dim1[y_index] = y0 + wy;
		
		cut->dim0[x_index] = x0;
		cut->dim1[x_index] = x0 + wx;
		
		cut->dim0[z_index] = z0;
		cut->dim1[z_index] = z0 + wz;
					
		COTraverser(tdata);
		
		return;
				
	}
	else if (wz == maxd && wz > S_Z_BASE) {
		
		cut->dim0[t_index] = t0;
		cut->dim1[t_index] = t0 + wt;
		
		
		cut->dim0[z_index] = z0;
		cut->dim1[z_index] = z0 + (wz >> 1);
		
		cut->dim0[x_index] = x0;
		cut->dim1[x_index] = x0 + wx;
		
		cut->dim0[y_index] = y0;
		cut->dim1[y_index] = y0 + wy;
		
		COTraverser(tdata);
		
		
		cut->dim0[t_index] = t0;
		cut->dim1[t_index] = t0 + wt;
		
		cut->dim0[z_index] = z0 + (wz >> 1);
		cut->dim1[z_index] = z0 + wz;
		
		cut->dim0[x_index] = x0;
		cut->dim1[x_index] = x0 + wx;
		
		cut->dim0[y_index] = y0;
		cut->dim1[y_index] = y0 + wy;
		
		COTraverser(tdata);
		
		
		return;		
	}
	else if (wt == maxd && wt > S_T_BASE) {
		
		cut->dim0[t_index] = t0;
		cut->dim1[t_index] = t0 + (wt >> 1);
		
		cut->dim0[x_index] = x0;
		cut->dim1[x_index] = x0 + wx;
		cut->dim0[y_index] = y0;
		cut->dim1[y_index] = y0 + wy;
		cut->dim0[z_index] = z0;
		cut->dim1[z_index] = z0 + wz;
		
		COTraverser(tdata);
		
		
		cut->dim0[t_index] = t0 + (wt >> 1);
		cut->dim1[t_index] = t0 + wt;
		
		cut->dim0[x_index] = x0;
		cut->dim1[x_index] = x0 + wx;
		cut->dim0[y_index] = y0;
		cut->dim1[y_index] = y0 + wy;
		cut->dim0[z_index] = z0;
		cut->dim1[z_index] = z0 + wz;
			
		
		COTraverser(tdata);
		
		return;
			
	} else {
		
		COComputeKernel(tdata);
	}
	
	//cout<<"finished"<<endl;
}
#endif
/**/
#if D3D
int get(int depth, int ycoor, int zcoor, int tcoor){

	
	int start = 0;
	for(int i = 1; i < depth; i++){
		int tmp = pow(2, i);
		start += pow(tmp, 3);
	}	
	int index = start + ycoor + zcoor * pow(2, depth) + tcoor * pow(4, depth);
	return index;
}
#else
int get(int depth, int xcoor, int ycoor, int zcoor, int tcoor){

	
	int index = offset[depth] + xcoor + ycoor * xarr_size[depth] + zcoor * xarr_size[depth]*yarr_size[depth] + tcoor * xarr_size[depth]* yarr_size[depth] * zarr_size[depth];
	
	return index;
}
#endif

#if D3D
void CollectAll(){
	int sum = 0;
	int counter = 0;
	for(int i = MIN_CUT-1; i >= 1; i--){
		sum = 0;
		for(int t = 0; t < pow(2, i); t++)
			for(int z = 0; z < pow(2, i); z++)
				for(int y = 0; y < pow(2, i); y++){
					int index = get(i, y, z, t);
					counter = lb_array[get(i+1, 2*y, 2*z, 2*t)] + lb_array[get(i+1, 2*y+1, 2*z, 2*t)] + lb_array[get(i+1, 2*y, 2*z+1, 2*t)] + lb_array[get(i+1, 2*y+1, 2*z+1, 2*t)] + lb_array[get(i+1, 2*y, 2*z, 2*t+1)] + lb_array[get(i+1, 2*y+1, 2*z, 2*t+1)] + lb_array[get(i+1, 2*y, 2*z+1, 2*t+1)] + lb_array[get(i+1, 2*y+1, 2*z+1, 2*t+1)]; 
					lb_array[index] = counter;
					sum += counter;
					
				}
				
		/*if(sum != 100000000){
			cout<<"here "<<i<<" "<<sum<<endl;
			char tt;
			cin>>tt;
		}*/ 
	}  
	
}
#else
void CollectAll(){
	
	int sum = 0;
	int counter = 0;
	for(int i = MIN_CUT4D-1; i >= 1; i--){
		sum = 0;
		for(int t = 0; t < pow(2, i); t++)
			for(int z = 0; z < pow(2, i); z++)
				for(int y = 0; y < pow(2, i); y++)
					for(int x = 0; x < pow(2, i); x++){
						int index = get(i, x, y, z, t);
						counter = lb_array4D[get(i+1, 2*x, 2*y, 2*z, 2*t)] + lb_array4D[get(i+1, 2*x+1, 2*y, 2*z, 2*t)] + lb_array4D[get(i+1, 2*x, 2*y+1, 2*z, 2*t)] 
								+ lb_array4D[get(i+1, 2*x+1, 2*y+1, 2*z, 2*t)] + lb_array4D[get(i+1, 2*x, 2*y, 2*z+1, 2*t)] + lb_array4D[get(i+1, 2*x+1, 2*y, 2*z+1, 2*t)] 
								+ lb_array4D[get(i+1, 2*x, 2*y+1, 2*z+1, 2*t)] + lb_array4D[get(i+1, 2*x+1, 2*y+1, 2*z+1, 2*t)] + lb_array4D[get(i+1, 2*x, 2*y, 2*z, 2*t+1)] 
								+ lb_array4D[get(i+1, 2*x+1, 2*y, 2*z, 2*t+1)] + lb_array4D[get(i+1, 2*x, 2*y+1, 2*z, 2*t+1)] + lb_array4D[get(i+1, 2*x+1, 2*y+1, 2*z, 2*t+1)] 
								+ lb_array4D[get(i+1, 2*x, 2*y, 2*z+1, 2*t+1)] + lb_array4D[get(i+1, 2*x+1, 2*y, 2*z+1, 2*t+1)] + lb_array4D[get(i+1, 2*x, 2*y+1, 2*z+1, 2*t+1)] 
								+ lb_array4D[get(i+1, 2*x+1, 2*y+1, 2*z+1, 2*t+1)]; 
						lb_array4D[index] = counter;
						sum += counter;
					
					}
				
				
		if(sum != (double)(SIZE_X*SIZE_Y*SIZE_Z*TIME_STEPS)){
			cout<<"here "<<i<<" "<<sum<<endl;
			char tt;
			cin>>tt;
		} 
	}  
	
}
void CheckAll(){
	
	double sum = 0;
	
	for(int i = 1; i <= MIN_CUT4D; i++){
		sum = 0;
		for(int t = offset[i]; t < offset[i] + tarr_size[i]*xarr_size[i]*yarr_size[i]*zarr_size[i]; t++){
			sum += lb_array4D[t]; 
		}
		if(sum != (double)(SIZE_X*SIZE_Y*SIZE_Z*TIME_STEPS)){
			cout<<"here "<<i<<" "<<sum<<endl;
			char tt;
			cin>>tt;
		} 
	}  
	
}
#endif
/**/
static bool Test2D(int t0, int t1, int y0, int y1){
	if((y0 - (t1-1)) > 0 &&  (y1-t0) <= SIZE_Y && t1 <= TIME_STEPS)
		return true;
		
	return false;	
}

static bool Test3D(int t0, int t1, int y0, int y1, int z0, int z1){
	if((y0 - (t1-1)) > 0 &&  (y1-t0) <= SIZE_Y && (z0 - (t1-1)) > 0 &&  (z1-t0) <= SIZE_Z && t1 <= TIME_STEPS)
		return true;
		
	return false;	
}

static bool Test4D(int t0, int t1, int x0, int x1, int y0, int y1, int z0, int z1){
	
		
	if((x0 - (t1-1)) >= 0 && (x1-t0) <= SIZE_X && (y0 - (t1-1)) > 0 &&  (y1-t0) <= SIZE_Y && (z0 - (t1-1)) > 0 &&  (z1-t0) <= SIZE_Z && t1 <= TIME_STEPS)
		return true;
		
	return false;	
}
#if _2D
static void LB_COComputeKernel(CO_THREAD_DATA *tdata, int* counter1){
	
	
	DOMAIN_3D_CUT *cut = &tdata->cut;
	int depth = tdata->depth;
	
		
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	int wt = t1 - t0;
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	int wx = x1 - x0;
	
	
	
	int counter = 0;
	
	
	if(Test2D(t0, t1, x0, x1)){
		counter = wx * wt; 
		
	}	
	else{
		for(int t = t0; t < min(t1, TIME_STEPS); t++) {
	
			counter += max((min((x1-t), SIZE_X) - max((x0-t), 0)), 0);
			
		}
	}
	int indices[4] = {0};
	
	indices[x_index] =  x0/xdiv_size[depth];
	indices[t_index] =  t0/tdiv_size[depth];
	
	int index = get(depth, indices[x_index], 0, 0, indices[t_index]);
	
	lb_array4D[index] = counter;
	
	*counter1 = counter;
	
}
#elif _3D
static void LB_COComputeKernel(CO_THREAD_DATA *tdata, int* counter1){
	
	
	DOMAIN_3D_CUT *cut = &tdata->cut;
	int depth = tdata->depth;
	
		
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	int wt = t1 - t0;
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	int wx = x1 - x0;
	
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	int wy = y1 - y0;
	
	//int z0 = cut->dim0[z_index];
	//int z1 = cut->dim1[z_index];
	//int wz = z1 - z0;
	
	
	int counter = 0;
	
	
	if(Test3D(t0, t1, x0, x1, y0, y1)){
		counter = wx * wy * wt; 
		
	}	
	else{
		for(int t = t0; t < min(t1, TIME_STEPS); t++) {
	
			counter += max((min((y1-t), SIZE_Y+1) - max((y0-t), 1)), 0) * max((min((x1-t), SIZE_X) - max((x0-t), 0)), 0);
			
		}
	}
	int indices[4] = {0};
	
	indices[x_index] =  x0/xdiv_size[depth];
	indices[y_index] =  y0/ydiv_size[depth];
	indices[t_index] =  t0/tdiv_size[depth];
	
	int index = get(depth, indices[x_index], indices[y_index], 0, indices[t_index]);
	
	lb_array4D[index] = counter;
	
	*counter1 = counter;
	
}
#else
static void LB_COComputeKernel(CO_THREAD_DATA *tdata, int* counter1){
	
	
	DOMAIN_3D_CUT *cut = &tdata->cut;
	int depth = tdata->depth;
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	int wt = t1 - t0;
	
	//if(wt == T)
	//	wt = TIME_STEPS;
		
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	int wx = x1 - x0;
	
	//if(wx == X)
	//	wx = SIZE_X;
		
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	int wy = y1 - y0;
	
	//if(wx == Y)
	//	wx = SIZE_Y;
		
	int z0 = cut->dim0[z_index];
	int z1 = cut->dim1[z_index];
	int wz = z1 - z0;
	
	//if(wx == Z)
	//	wx = SIZE_Z;
	
	int counter = 0;
	
	//if(t1 < 100 && y1 < 100 && z1 < 100)
	//	cout<<t0<<" "<<t1<<" "<<y0<<" "<<y1<<" "<<z0<<" "<<z1<<endl;
	
	//if(t0 >= TIME_STEPS)
	//	return;
	
	if(Test4D(t0, t1, x0, x1, y0, y1, z0, z1)){
		//cout<<"here"<<endl;
		counter = (wx) * (wy) * (wt) * (wz); 
		//cout<<counter<<endl;
	}	
	else{
		for(int t = t0; t < min(t1, TIME_STEPS); t++) {
	
			/*for(int i = max((z0-t), 1); i < min((z1-t), SIZE_Z+1); i++){
			for(int j = max((y0-t), 1); j < min((y1-t), SIZE_Y+1); j++){
			for(int k = max((x0-t), 0); k < min((x1-t), SIZE_X); k++){	/**/
			//counter++;
			
			counter += max((min((z1-t), SIZE_Z+1) - max((z0-t), 1)), 0) * max((min((y1-t), SIZE_Y+1) - max((y0-t), 1)), 0) * max((min((x1-t), SIZE_X) - max((x0-t), 0)), 0);
					
				
		}//}}}
	}
		
	int indices[4] = {0};
	
	//if(xdiv_size[depth] > 1){
		indices[x_index] = x0/xdiv_size[depth];
	//}
	//if(ydiv_size[depth] > 1){
		indices[y_index] = y0/ydiv_size[depth];
	//}
	//if(zdiv_size[depth] > 1){
		indices[z_index] = z0/zdiv_size[depth];
	//}
	//if(tdiv_size[depth] > 1){
		indices[t_index] = t0/tdiv_size[depth];
	//}
	
	
	int index = get(depth, indices[x_index], indices[y_index], indices[z_index], indices[t_index]);
	/*if(tdata->id == 0){
		cout<<xarr_size[depth]<<" "<<yarr_size[depth]<<" "<<zarr_size[depth]<<" "<<t0/tdiv_size[depth]<<endl;
		cout<<x0<<" "<<y0<<" "<<z0<<" "<<t0<<endl;
		cout<<index<<endl;
		cout<<offset[depth]<<endl;
	}*/
	lb_array4D[index] = counter;
	
	*counter1 = counter;
	
}
static void LB_COComputeKernel3D(CO_THREAD_DATA *tdata, int ind1, int ind2, int ind3, int ind4){
	//put wdim[ind4] = some size
	DOMAIN_3D_CUT *cut = &tdata->cut;
	int depth = tdata->depth;
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	int wt = t1 - t0;
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	int wx = x1 - x0;
	
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	int wy = y1 - y0;
	
	int z0 = cut->dim0[z_index];
	int z1 = cut->dim1[z_index];
	int wz = z1 - z0;
	
	
	int counter = 0;
	
	//if(t1 < 100 && y1 < 100 && z1 < 100)
	//	cout<<t0<<" "<<t1<<" "<<y0<<" "<<y1<<" "<<z0<<" "<<z1<<endl;
	
	//if(t0 >= TIME_STEPS)
	//	return;
	if(Test4D(t0, t1, x0, x1, y0, y1, z0, z1)){
		
		counter = (wx) * (wy) * (wt) * (wz); 
		//cout<<counter<<endl;
	}	
	else{
		for(int t = t0; t < min(t1, TIME_STEPS); t++) {
	
			for(int i = max((z0-t), 1); i < min((z1-t), SIZE_Z+1); i++){
			for(int j = max((y0-t), 1); j < min((y1-t), SIZE_Y+1); j++){
			for(int k = max((x0-t), 0); k < min((x1-t), SIZE_X); k++){	
			//for(int k = max((x0-t), 0); k < min((x1-t), SIZE_X-1); k++){
						
				//if(t < TIME_STEPS && i <= SIZE_Z && i > 0 && j > 0 && j <= SIZE_Y && k < SIZE_X && k >= 0){
					//cout<<"here"<<endl;		
					counter++;
					
				//}
			}}}//}
		}
	}
	int div = pow(2, depth);
	int div_size[4];
	int ind[4];
	div_size[t_index] = tdiv_size[0]/div;
	div_size[x_index] = xdiv_size[0]/div;
	div_size[y_index] = ydiv_size[0]/div;
	div_size[z_index] = zdiv_size[0]/div;
	
	ind[t_index] = t0/div_size[t_index];
	ind[x_index] = x0/div_size[x_index];
	ind[y_index] = y0/div_size[y_index];
	ind[z_index] = z0/div_size[z_index];
	ind[ind4] = 0;
	
	int index = get(depth, ind[x_index], ind[y_index], ind[z_index], ind[t_index]);
	
	lb_array4D[index] = counter;
	
}
#endif

#if _3D
int Stest(S_DOMAIN_3D_CUT *c){
	
	int x0 = c->dim0[x_index];
	int wx = c->dim1[x_index] - x0;
	
	int y0 = c->dim0[y_index];
	int wy = c->dim1[y_index] - y0;
	
	int t0 = c->dim0[t_index];
	int h = c->dim1[t_index] - t0;
	
	int s_diff1 = x0-t0;
	int s_diff2 = s_diff1 + wx-1;
	int s_diff3 = y0-t0;
	int s_diff4 = s_diff3 + wy-1;
	
	if(s_diff2 <= 0 || s_diff4 <= 0)
		return 0;
	
	if((s_diff1-h+1) > SIZE_X || (s_diff3-h+1) > SIZE_Y)
		return 0; 	
	
	return 1;
}

int Ttest(S_DOMAIN_3D_CUT *c){
	
	int t0 = c->dim0[t_index];
	if(t0 > TIME_STEPS - 1)
		return 0;
	
	return Stest(c);	
}
#elif _2D
int Stest(S_DOMAIN_3D_CUT *c){
	
	int x0 = c->dim0[x_index];
	int wx = c->dim1[x_index] - x0;
	
		
	int t0 = c->dim0[t_index];
	int h = c->dim1[t_index] - t0;
	
	int s_diff1 = x0-t0;
	int s_diff2 = s_diff1 + wx-1;
	
	
	if(s_diff2 <= 0)
		return 0;
	
	if((s_diff1-h+1) > SIZE_X )
		return 0; 	
	
	return 1;
}

int Ttest(S_DOMAIN_3D_CUT *c){
	
	int t0 = c->dim0[t_index];
	if(t0 > TIME_STEPS - 1)
		return 0;
	
	return Stest(c);	
}
#else
int Stest(S_DOMAIN_3D_CUT *c){
	
	int x0 = c->dim0[x_index];
	int wx = c->dim1[x_index] - x0;
	
	int y0 = c->dim0[y_index];
	int wy = c->dim1[y_index] - y0;
	
	int z0 = c->dim0[z_index];
	int wz = c->dim1[z_index] - z0;
	
	int t0 = c->dim0[t_index];
	int h = c->dim1[t_index] - t0;
	
	int s_diff1 = y0-t0;
	int s_diff2 = s_diff1 + wy-1;
	int s_diff3 = z0-t0;
	int s_diff4 = s_diff3 + wz-1;
	int s_diff5 = x0-t0;
	int s_diff6 = s_diff5 + wx-1;
	
	if(s_diff2 <= 0 || s_diff4 <= 0 || s_diff6 < 0)
		return 0;
	
	if((s_diff1-h+1) > SIZE_Y || (s_diff3-h+1) > SIZE_Z || (s_diff5-h+1) > SIZE_X-1)
		return 0; 	
	
	return 1;
}

int Ttest(S_DOMAIN_3D_CUT *c){
	
	int t0 = c->dim0[t_index];
	if(t0 > TIME_STEPS - 1)
		return 0;
	
	return Stest(c);	
}
#endif
#if D3D
void LB_Calc(int depth, DOMAIN_3D_CUT* cut1, float* tn1, float* tn2, float tn){
	
	int div = pow(2, depth+1);
	int tdiv_size = T/div;
	int ydiv_size = Y/div;
	int zdiv_size = Z/div;
	
	int y11 = cut1[0].dim1[y_index];
	int z11 = cut1[0].dim1[z_index];
	int t11 = cut1[0].dim1[t_index];
	
	int y21 = cut1[1].dim1[y_index];
	int z21 = cut1[1].dim1[z_index];
	int t21 = cut1[1].dim1[t_index];
	
	int index1 = get(depth+1, (y11/ydiv_size) - 1, (z11/zdiv_size) - 1, (t11/tdiv_size) - 1);
	int index2 = get(depth+1, (y21/ydiv_size) - 1, (z21/zdiv_size) - 1, (t21/tdiv_size) - 1);
		
	float a = ((float)lb_array[index1]/(float)(lb_array[index1] + (float)lb_array[index2]))*tn;
		
	float tf = floor(a);
	float tc = ceil(a);
	
	float pf = ((tf == 0.0 ) ? epsilon : a/tf);
	float pc = ((tc == tn ) ? epsilon : (tn - a)/(tn - tc));
	
	(*tn1) = (pf < pc) ? tf : tc;
	(*tn2) = tn - (*tn1);
	
	if((*tn1) == 0 && a > 0)
		(*tn1) = tn;
	if((*tn2) == 0 && a < tn)
		(*tn2) = tn;
}
#else
void LB_Calc(int depth, DOMAIN_3D_CUT* cut1, float* tn1, float* tn2, float tn){
	
	
	
	int x10 = cut1[0].dim0[x_index];
	int y10 = cut1[0].dim0[y_index];
	int z10 = cut1[0].dim0[z_index];
	int t10 = cut1[0].dim0[t_index];
	
	int x20 = cut1[1].dim0[x_index];
	int y20 = cut1[1].dim0[y_index];
	int z20 = cut1[1].dim0[z_index];
	int t20 = cut1[1].dim0[t_index];
	
	int indices1[4] = {0};
	int indices2[4] = {0};
	
	
	indices1[x_index] = x10/xdiv_size[depth+1];
	indices2[x_index] = x20/xdiv_size[depth+1];	
	
	indices1[y_index] = y10/ydiv_size[depth+1];
	indices2[y_index] = y20/ydiv_size[depth+1];
	
	indices1[z_index] = z10/zdiv_size[depth+1];
	indices2[z_index] = z20/zdiv_size[depth+1];
	
	indices1[t_index] = t10/tdiv_size[depth+1];
	indices2[t_index] = t20/tdiv_size[depth+1];
	
	
	
	int index1 = get(depth+1, indices1[x_index], indices1[y_index], indices1[z_index], indices1[t_index]);
	int index2 = get(depth+1, indices2[x_index], indices2[y_index], indices2[z_index], indices2[t_index]);
		
	float a = ((float)lb_array4D[index1]/(float)(lb_array4D[index1] + (float)lb_array4D[index2]))*tn;
		
	float tf = floor(a);
	float tc = ceil(a);
	
	float pf = ((tf == 0.0 ) ? epsilon : a/tf);
	float pc = ((tc == tn ) ? epsilon : (tn - a)/(tn - tc));
	
	(*tn1) = (pf < pc) ? tf : tc;
	(*tn2) = tn - (*tn1);
	
	if((*tn1) == 0 && a > 0)
		(*tn1) = tn;
	if((*tn2) == 0 && a < tn)
		(*tn2) = tn;
		
	
	
}

#endif

#ifndef D3D 
//1,4,6,4,1 division

static void distribute(CO_THREAD_DATA *threadData, DOMAIN_3D_CUT* cut1){
	
	int tstart = threadData->tstart;
	int tend = threadData->tend;
	int depth = threadData->depth;
	
	float tn = (float)(tend-tstart+1.0);
	float tn1, tn2;
#if LB
	LB_Calc(depth, cut1, &tn1, &tn2, tn);
		
	if(tn1 != 0.0 && threadData->id >= tstart && threadData->id <= (tstart+tn1-1)){
	
		threadData->tstart = tstart;
		threadData->tend = tstart+tn1-1;
		
		threadData->cut = cut1[0];
		
		threadData->depth++;
		PCOTraverser(threadData);
		threadData->depth--;
	}	
	if(tn2 != 0.0 && threadData->id >= (tend-tn2+1) && threadData->id <= tend){
		
		threadData->tstart = (tend - tn2 + 1);
		threadData->tend = tend;
		
		threadData->cut = cut1[1];
		threadData->depth++;
		PCOTraverser(threadData);
		threadData->depth--;
	}		
	
#else
	
	int tsplit = (tstart + tend) >> 1;
		
	if(threadData->id <= tsplit){
		
		threadData->tstart = tstart;
		threadData->tend = tsplit;
		
		threadData->cut = cut1[0];
	}		
	else{
		threadData->tstart = (tsplit + 1);
		threadData->tend = tend;
		
		threadData->cut = cut1[1];
		
	}
		
	threadData->depth++;
	PCOTraverser(threadData);
	threadData->depth--;
#endif	

	threadData->tstart = tstart;
	threadData->tend = tend;
	threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);		

}

#ifdef NLB

/*
 * Calculate number of points in each tile and store them into 'points' array
 * points array should be allocated before invocation of calculatePoints
 */
void calculatePoints(int depth, DOMAIN_3D_CUT* cut1, int nTiles, int *points){

    for(int i = 0; i < nTiles; i++) {
        int xi0 = cut1[i].dim0[x_index];
        int yi0 = cut1[i].dim0[y_index];
        int zi0 = cut1[i].dim0[z_index];
        int ti0 = cut1[i].dim0[t_index];

        int indices[4];

        indices[x_index] = xi0/xdiv_size[depth+1];
	indices[y_index] = yi0/ydiv_size[depth+1];
	indices[z_index] = zi0/zdiv_size[depth+1];
	indices[t_index] = ti0/tdiv_size[depth+1];

        int index = get(depth+1, indices[x_index], indices[y_index], indices[z_index], indices[t_index]);

        points[i] = lb_array4D[index];
    }
}

static void distribute(CO_THREAD_DATA *threadData, DOMAIN_3D_CUT* cut1, int nCuts){

	int tstart = threadData->tstart;
	int tend = threadData->tend;
	int depth = threadData->depth;

	int tn = tend - tstart + 1; // Number of available threads

        int points = new int[nCuts];

	calculatePoints(depth, cut1, nCuts, points);

        AllocationTable *allocationTable = balanceLoad(points, nCuts, tn);
        AllocationTable *currentRow = allocationTable;

        delete []points;

        while(currentRow) {
            int start;
            int end = tstart;
            for(int i = 0; i < nCuts; i++) {
                start = end;
                end = start + currentRow -> allocationList[i];
                if (currentRow -> allocationList[i] && threadData->id >= start && threadData->id < end) {
                    threadData->tstart = start;
                    threadData->tend = end - 1;

                    threadData->cut = cut1[i];

                    threadData->depth++;
                    PCOTraverser(threadData);
                    threadData->depth--;
                }
            }
            threadData->tstart = tstart;
            threadData->tend = tend;
            threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);

            currentRow = currentRow -> nextList;
        }

        delete allocationTable;
}


/*
 * dimention: either 3D or 4D
 *  For 3D, the function extract independent cuts based on 1,3,3,1
 *  For 4D, the function extract independent cuts based on 1,4,6,4,1
 * level: index of Pascal Triangle element starting from 0, e.g. 6 in 4D case (1,4,6,4,1) has index 2
 */
static DOMAIN_3D_CUT* findIndependentCuts(int dimention, int level, int pind[], int dim0[], int dim1[], int wdim[], int& nCuts) {
    DOMAIN_3D_CUT *independentCuts = NULL;

    switch(dimention) {
        case 3:
            if(level == 1) {
                nCuts = 3;
                independentCuts = new DOMAIN_3D_CUT[nCuts];

                /*
                 * Independent tiles: 001, 010, 100
                 */

                // 001
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 010
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 100
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                return independentCuts;

            }
            else if (level == 2) {
                nCuts = 3;
                independentCuts = new DOMAIN_3D_CUT[nCuts];

                /*
                 * Independent tiles: 011, 101, 110
                 */

                // 011
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 101
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 110
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                return independentCuts;
            }
            else {
                nCuts = 0;
                cerr<<"Not supported level "<<level<<" in dimention "<<dimention<<endl;
                return NULL;
            }
            break;
        case 4:
            if(level == 1) {
                nCuts = 4;
                independentCuts = new DOMAIN_3D_CUT[nCuts];

                /*
                 * Independent tiles: 0001, 0010, 0100, 1000
                 */

                // 0001
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 0010
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 0100
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 1000
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);
                independentCuts[3].dim1[pind[3]] = dim1[pind[3]];

                return independentCuts;
            }
            else if (level == 2) {
                nCuts = 6;
                independentCuts = new DOMAIN_3D_CUT[nCuts];

                /*
                 * Independent tiles: 0011, 0101, 0110, 1001, 1010, 1100
                 */

                // 0011
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 0101
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 0110
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 1001
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);
                independentCuts[3].dim1[pind[3]] = dim1[pind[3]];

                // 1010
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);
                independentCuts[3].dim1[pind[3]] = dim1[pind[3]];

                // 1100
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);
                independentCuts[3].dim1[pind[3]] = dim1[pind[3]];

                return independentCuts;
            }
            else if (level == 3) {
                nCuts = 4;
                independentCuts = new DOMAIN_3D_CUT[nCuts];

                /*
                 * Independent tiles: 0111, 1011, 1101, 1110
                 */

                // 0111
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]];
                independentCuts[3].dim1[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);

                // 1011
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]];
                independentCuts[2].dim1[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);
                independentCuts[3].dim1[pind[3]] = dim1[pind[3]];

                // 1101
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[0].dim1[pind[0]] = dim1[pind[0]];
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]];
                independentCuts[1].dim1[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);
                independentCuts[3].dim1[pind[3]] = dim1[pind[3]];

                // 1110
                independentCuts[0].dim0[pind[0]] = dim0[pind[0]];
                independentCuts[0].dim1[pind[0]] = dim0[pind[0]] + (wdim[pind[0]] >> 1);
                independentCuts[1].dim0[pind[1]] = dim0[pind[1]] + (wdim[pind[1]] >> 1);
                independentCuts[1].dim1[pind[1]] = dim1[pind[1]];
                independentCuts[2].dim0[pind[2]] = dim0[pind[2]] + (wdim[pind[2]] >> 1);
                independentCuts[2].dim1[pind[2]] = dim1[pind[2]];
                independentCuts[3].dim0[pind[3]] = dim0[pind[3]] + (wdim[pind[3]] >> 1);
                independentCuts[3].dim1[pind[3]] = dim1[pind[3]];

                return independentCuts;
            }
            else {
                nCuts = 0;
                cerr<<"Not supported level "<<level<<" in dimention "<<dimention<<endl;
                return NULL;
            }
            break;
        default:
            cerr<<"Not supported dimention: "<<dimention<<endl;
    }
}

#endif
static void PCOTraverser(CO_THREAD_DATA *threadData) {
	
	//cout<<"actual computation started"<<endl;
	int tstart = threadData->tstart;
	int tend = threadData->tend;
	
	
	int pind[DIM] = {0};
	int ind1 = 0;
	int ind2 = 0;
	int ind3 = 0;
	int ind4 = 0;
	int caseNr = 0;
	int maxSize = 1;
	float tn1, tn2;
	int j = 0;
	int i = 0;		                      
	int pairs4D[]= {0b0001, 0b0010, 0b0100, 0b1000, 0b0101, 0b0110, 0b0011, 0b1001, 0b1100, 0b1010, 0b1101, 0b1011, 0b0111, 0b1110}; 
	int pairs3D[]= {0b001, 0b010, 0b100, 0b011, 0b110, 0b101}; 
	int pairs2D[] = {0b01, 0b10};
	
		
	int dim0[DIM];
	int dim1[DIM];
	int wdim[DIM];
	
	DOMAIN_3D_CUT cut1[2];
	DOMAIN_3D_CUT *cut = &threadData->cut;
	
	dim0[t_index] = cut->dim0[t_index];
	dim1[t_index] = cut->dim1[t_index];
	
	dim0[x_index] = cut->dim0[x_index];
	dim1[x_index] = cut->dim1[x_index];
	
	dim0[y_index] = cut->dim0[y_index];
	dim1[y_index] = cut->dim1[y_index];
	
	dim0[z_index] = cut->dim0[z_index];
	dim1[z_index] = cut->dim1[z_index];
	
	wdim[t_index] = cut->dim1[t_index] - dim0[t_index];
	wdim[x_index] = cut->dim1[x_index] - dim0[x_index];
	wdim[y_index] = cut->dim1[y_index] - dim0[y_index];
	wdim[z_index] = cut->dim1[z_index] - dim0[z_index];
	
	
	// input values: current tileSize and parallel baseSize
	int relSize[]= {
   		wdim[t_index]/M_BASE[t_index],
   		wdim[x_index]/M_BASE[x_index],
	   	wdim[y_index]/M_BASE[y_index],
	   	wdim[z_index]/M_BASE[z_index],
	};
	
		
	// choose scheduler
	
	
	switch( CURRENT_SCHEDULER ) {
		
			
		case SCHEDULER_PARALLELISM:
		   
		   for(i = 0; i < DIM; i++) {
		     if(relSize[i] > 1) {
		       pind[caseNr] = i; 
		       caseNr++;
		     }
		   }
		   j = caseNr;
		   for(i = 0; i < DIM; i++){
		   	if(relSize[i] <= 1){
		     	pind[j++] = i; 
		     }
		   }
		   break;
		
		case SCHEDULER_FORM:
		   
		   for(i = 0; i < DIM; i++) {
		     if(maxSize<relSize[i]) 
		     	maxSize = relSize[i];
		   }
		   
		   for(i = 0; i < DIM; i++) {
		   			   	 
		   	if(relSize[i] == maxSize) {
			
		       pind[caseNr] = i;
		       caseNr++;
		    }
		   }
		   j = caseNr;
		   for(i = 0; i < DIM; i++){
		   	if(relSize[i] != maxSize){
		     	pind[j++] = i; 
		     }
		   }
		   if(maxSize == 1) 
		   	caseNr= 0;
		   break;
	}
	
		
 	ind1 = pind[0];
	ind2 = pind[1];
	ind3 = pind[2];
	ind4 = pind[3];
	
	if (tstart == tend) {
		
		COTraverser(threadData);
		return;
	}

	
	
	
	
 	
 	switch(caseNr){
 		
 		
		
 		case 4:
	 		
	 		//cout<<"entered case 4 id: "<<threadData->id<<endl; 		
 			cut->dim0[ind1] = dim0[ind1];
			cut->dim1[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
			cut->dim0[ind2] = dim0[ind2];
			cut->dim1[ind2] = dim0[ind2] + (wdim[ind2] >> 1);
			cut->dim0[ind3] = dim0[ind3];
			cut->dim1[ind3] = dim0[ind3] + (wdim[ind3] >> 1);
			cut->dim0[ind4] = dim0[ind4];
			cut->dim1[ind4] = dim0[ind4] + (wdim[ind4] >> 1);
	
	
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
	
			threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);

#ifndef NLB // Old Load Balancer
			
			for(int i = 0; i < 7; i++){
	
				
				cut1[0].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs4D[2*i]&(1<<0))));
				cut1[0].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((15-pairs4D[2*i])&(1<<0))));
				cut1[0].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs4D[2*i]&(1<<1))));
				cut1[0].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((15-pairs4D[2*i])&(1<<1))));
				cut1[0].dim0[ind3] = dim0[ind3] + ((wdim[ind3] >> 1) * bool((pairs4D[2*i]&(1<<2))));
				cut1[0].dim1[ind3] = dim1[ind3] - ((wdim[ind3] >> 1) * bool(((15-pairs4D[2*i])&(1<<2))));
				cut1[0].dim0[ind4] = dim0[ind4] + ((wdim[ind4] >> 1) * bool((pairs4D[2*i]&(1<<3))));
				cut1[0].dim1[ind4] = dim1[ind4] - ((wdim[ind4] >> 1) * bool(((15-pairs4D[2*i])&(1<<3))));
	
				
				cut1[1].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs4D[2*i+1]&(1<<0))));
				cut1[1].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((15-pairs4D[2*i+1])&(1<<0))));
				cut1[1].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs4D[2*i+1]&(1<<1))));
				cut1[1].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((15-pairs4D[2*i+1])&(1<<1))));
				cut1[1].dim0[ind3] = dim0[ind3] + ((wdim[ind3] >> 1) * bool((pairs4D[2*i+1]&(1<<2))));
				cut1[1].dim1[ind3] = dim1[ind3] - ((wdim[ind3] >> 1) * bool(((15-pairs4D[2*i+1])&(1<<2))));
				cut1[1].dim0[ind4] = dim0[ind4] + ((wdim[ind4] >> 1) * bool((pairs4D[2*i+1]&(1<<3))));
				cut1[1].dim1[ind4] = dim1[ind4] - ((wdim[ind4] >> 1) * bool(((15-pairs4D[2*i+1])&(1<<3))));
	
				if(Ttest(&cut1[0]) || Ttest(&cut1[1])){
		
					distribute(threadData, cut1);					

				}
	
			}
#else // New Load Balancer
                        // in (1, 4, 6, 4, 1), (4, 6, 4) should be produced
                        for (int i = 1; i < 4; i++) {
                            int nCuts;
                            DOMAIN_3D_CUT *independentCuts = findIndependentCuts(4, i, pind, dim0, dim1, wdim, nCuts);

                            bool areCutsOk = false;
                            for(int j = 0; j < nCuts && !areCutsOk; j++) {
                                areCutsOk = areCutsOk || Ttest(&independentCuts[j]);
                            }

                            // If cuts are Ok => ditribute thread over cuts
                            if(areCutsOk)
                                distribute(threadData, independentCuts, nCuts);

                            delete []independentCuts;
                        }
#endif
			
			cut->dim0[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
			cut->dim1[ind1] = dim1[ind1];
			cut->dim0[ind2] = dim0[ind2] + (wdim[ind2] >> 1);
			cut->dim1[ind2] = dim1[ind2];
			cut->dim0[ind3] = dim0[ind3] + (wdim[ind3] >> 1);
			cut->dim1[ind3] = dim1[ind3];
			cut->dim0[ind4] = dim0[ind4] + (wdim[ind4] >> 1);
			cut->dim1[ind4] = dim1[ind4];
	

			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
		break;
		
		case 3:
			//cout<<"entered case 3 id: "<<threadData->id<<endl;
			/*if (wdim[ind1] <= M_BASE[ind1] || wdim[ind2] <= M_BASE[ind2] || wdim[ind3] <= M_BASE[ind3]) {
				
				if (threadData->id == tstart) {
					COTraverser(threadData);
				}
				return;
				
			}*/		
			
			
			cut->dim0[ind1] = dim0[ind1];
			cut->dim1[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
			cut->dim0[ind2] = dim0[ind2];
			cut->dim1[ind2] = dim0[ind2] + (wdim[ind2] >> 1);
			cut->dim0[ind3] = dim0[ind3];
			cut->dim1[ind3] = dim0[ind3] + (wdim[ind3] >> 1);
			cut->dim0[ind4] = dim0[ind4];
			cut->dim1[ind4] = dim1[ind4];
	
	
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
	
			threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);
#ifndef NLB // Old Load Balancer
			for(int i = 0; i < 3; i++){
	
				
				cut1[0].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs3D[2*i]&(1<<0))));
				cut1[0].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((7-pairs3D[2*i])&(1<<0))));
				cut1[0].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs3D[2*i]&(1<<1))));
				cut1[0].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((7-pairs3D[2*i])&(1<<1))));
				cut1[0].dim0[ind3] = dim0[ind3] + ((wdim[ind3] >> 1) * bool((pairs3D[2*i]&(1<<2))));
				cut1[0].dim1[ind3] = dim1[ind3] - ((wdim[ind3] >> 1) * bool(((7-pairs3D[2*i])&(1<<2))));
				cut1[0].dim0[ind4] = dim0[ind4];
				cut1[0].dim1[ind4] = dim1[ind4];
				
	
				
				cut1[1].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs3D[2*i+1]&(1<<0))));
				cut1[1].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((7-pairs3D[2*i+1])&(1<<0))));
				cut1[1].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs3D[2*i+1]&(1<<1))));
				cut1[1].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((7-pairs3D[2*i+1])&(1<<1))));
				cut1[1].dim0[ind3] = dim0[ind3] + ((wdim[ind3] >> 1) * bool((pairs3D[2*i+1]&(1<<2))));
				cut1[1].dim1[ind3] = dim1[ind3] - ((wdim[ind3] >> 1) * bool(((7-pairs3D[2*i+1])&(1<<2))));
				cut1[1].dim0[ind4] = dim0[ind4];
				cut1[1].dim1[ind4] = dim1[ind4];
	
				if(Ttest(&cut1[0]) || Ttest(&cut1[1])){
		
					distribute(threadData, cut1);					

				}
	
			}
#else // New Load Balancer
                        // in (1, 3, 3, 1), (3,3) should be produced
                        for (int i = 1; i < 3; i++) {
                            int nCuts;
                            DOMAIN_3D_CUT *independentCuts = findIndependentCuts(3, i, pind, dim0, dim1, wdim, nCuts);

                            bool areCutsOk = false;
                            for(int j = 0; j < nCuts && !areCutsOk; j++) {
                                areCutsOk = areCutsOk || Ttest(&independentCuts[j]);
                            }

                            // If cuts are Ok => ditribute thread over cuts
                            if(areCutsOk)
                                distribute(threadData, independentCuts, nCuts);

                            delete []independentCuts;
                        }
#endif
			
			cut->dim0[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
			cut->dim1[ind1] = dim1[ind1];
			cut->dim0[ind2] = dim0[ind2] + (wdim[ind2] >> 1);
			cut->dim1[ind2] = dim1[ind2];
			cut->dim0[ind3] = dim0[ind3] + (wdim[ind3] >> 1);
			cut->dim1[ind3] = dim1[ind3];
			cut->dim0[ind4] = dim0[ind4];
			cut->dim1[ind4] = dim1[ind4];
			
	

			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
	
			
		break;
		
		case 2:
			//cout<<"entered case 2 id: "<<threadData->id<<endl;
			/*if (wdim[ind1] <= M_BASE[ind1] || wdim[ind2] <= M_BASE[ind2]) {
		
				if (threadData->id == tstart) {
					COTraverser(threadData);
				}
				return;
				
			}*/
						
			cut->dim0[ind1] = dim0[ind1];
			cut->dim1[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
			cut->dim0[ind2] = dim0[ind2];
			cut->dim1[ind2] = dim0[ind2] + (wdim[ind2] >> 1);
			cut->dim0[ind3] = dim0[ind3];
			cut->dim1[ind3] = dim1[ind3];
			cut->dim0[ind4] = dim0[ind4];
			cut->dim1[ind4] = dim1[ind4];
	
	
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
	
			threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);
			
			
			cut1[0].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs2D[0]&(1<<0))));
			cut1[0].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((3-pairs2D[0])&(1<<0))));
			cut1[0].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs2D[0]&(1<<1))));
			cut1[0].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((3-pairs2D[0])&(1<<1))));
			cut1[0].dim0[ind3] = dim0[ind3];
			cut1[0].dim1[ind3] = dim1[ind3];
			cut1[0].dim0[ind4] = dim0[ind4];
			cut1[0].dim1[ind4] = dim1[ind4];
			
	
			cut1[1].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs2D[1]&(1<<0))));
			cut1[1].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((3-pairs2D[1])&(1<<0))));
			cut1[1].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs2D[1]&(1<<1))));
			cut1[1].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((3-pairs2D[1])&(1<<1))));
			cut1[1].dim0[ind3] = dim0[ind3];
			cut1[1].dim1[ind3] = dim1[ind3];
			cut1[1].dim0[ind4] = dim0[ind4];
			cut1[1].dim1[ind4] = dim1[ind4];
	
			if(Ttest(&cut1[0]) || Ttest(&cut1[1])){
		
				distribute(threadData, cut1);			

			}
	
			
			
			cut->dim0[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
			cut->dim1[ind1] = dim1[ind1];
			cut->dim0[ind2] = dim0[ind2] + (wdim[ind2] >> 1);
			cut->dim1[ind2] = dim1[ind2];
			cut->dim0[ind3] = dim0[ind3];
			cut->dim1[ind3] = dim1[ind3];
			cut->dim0[ind4] = dim0[ind4];
			cut->dim1[ind4] = dim1[ind4];
			
	

			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
			
		break;
		
		case 1:
			//cout<<"entered case 1 id: "<<threadData->id<<endl;
			/*if (wdim[ind1] <= M_BASE[ind1]) {
		
				if (threadData->id == tstart) {
					COTraverser(threadData);
				}
				return;
			}*/
			
			cut->dim0[ind1] = dim0[ind1];
			cut->dim1[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
			cut->dim0[ind2] = dim0[ind2];
			cut->dim1[ind2] = dim1[ind2];
			cut->dim0[ind3] = dim0[ind3];
			cut->dim1[ind3] = dim1[ind3];
			cut->dim0[ind4] = dim0[ind4];
			cut->dim1[ind4] = dim1[ind4];
	
	
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
	
			threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);
						
			
			cut->dim0[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
			cut->dim1[ind1] = dim1[ind1];
			cut->dim0[ind2] = dim0[ind2];
			cut->dim1[ind2] = dim1[ind2];
			cut->dim0[ind3] = dim0[ind3];
			cut->dim1[ind3] = dim1[ind3];
			cut->dim0[ind4] = dim0[ind4];
			cut->dim1[ind4] = dim1[ind4];
			
	

			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
			
		break;
		
		case 0:
			
			if (threadData->id == tstart) {
				//cout<<"entered case 0 id: "<<threadData->id<<endl;
				
				COTraverser(threadData);
			}
			return;
			break;
			
	
 	}
}

#else
void PCOTraverser(CO_THREAD_DATA *threadData, int dim) {//multithreaded Cache Oblivious
	//cout<<lb_array[1]<<" "<<lb_array[4]<<endl;
	//char ttt;
	//cin>>ttt;
	
	//cout<<lb_array[1]<<endl;
	//char y;
	//cin>>y;
	int tstart = threadData->tstart;
	int tend = threadData->tend;
	float tn = (float)(tend-tstart+1.0);
	int depth = threadData->depth;
	
				
	float tn1, tn2;
	
	
	DOMAIN_3D_CUT *cut = &threadData->cut;
	
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	
	int z0 = cut->dim0[z_index];
	int z1 = cut->dim1[z_index];
	
	int wt = cut->dim1[t_index] - t0;
	int wy = cut->dim1[y_index] - y0;
	int wz = cut->dim1[z_index] - z0;
	
	
	if (tstart == tend) {
		COTraverser(threadData);
		return;
	}

	
	if (wt <= M_T_BASE || wy <= M_Y_BASE || wz <= M_Z_BASE) {
		//cout<<"left from here"<<endl;
		
		if (threadData->id == tstart) {
			PerfTimer timer;
			timer.start();
			COTraverser(threadData);
			accum += (tn-1)*(timer.elapsed());
			//delete timer;
		}
		return;
	}
	//000
	cut->dim0[t_index] = t0;
	cut->dim1[t_index] = t0 + (wt >> 1);
	cut->dim0[y_index] = y0;
	cut->dim1[y_index] = y0 + (wy >> 1);
	cut->dim0[z_index] = z0;
	cut->dim1[z_index] = z0 + (wz >> 1);
	
	if(Ttest(cut)){
		threadData->depth++;
		PCOTraverser(threadData);
		threadData->depth--;
	}
	
	
	threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);
	
	DOMAIN_3D_CUT cut1[2];
	
	
			
	//z
	cut1[0].dim0[t_index] = t0;
	cut1[0].dim1[t_index] = t0 + (wt >> 1);
	cut1[0].dim0[x_index] = x0;
	cut1[0].dim1[x_index] = x1;
	cut1[0].dim0[y_index] = y0;
	cut1[0].dim1[y_index] = y0 + (wy >> 1);
	cut1[0].dim0[z_index] = z0 + (wz >> 1);
	cut1[0].dim1[z_index] = z1;
	
	//y
	cut1[1].dim0[t_index] = t0;
	cut1[1].dim1[t_index] = t0 + (wt >> 1);
	cut1[1].dim0[x_index] = x0;
	cut1[1].dim1[x_index] = x1;
	cut1[1].dim0[y_index] = y0 + (wy >> 1);
	cut1[1].dim1[y_index] = y1;
	cut1[1].dim0[z_index] = z0;
	cut1[1].dim1[z_index] = z0 + (wz >> 1);
	
		
	
	if(Ttest(&cut1[0]) || Ttest(&cut1[1])){
	
#if LB
		LB_Calc(depth, cut1, &tn1, &tn2, tn);
		
				
		if(tn1 != 0 && threadData->id >= tstart && threadData->id <= (tstart+tn1-1)){
		
			threadData->tstart = tstart;
			threadData->tend = tstart+tn1-1;
			//z
			threadData->cut = cut1[0];
		
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
					
		}
		if(tn2 != 0 && threadData->id >= (tend-tn2+1) && threadData->id <= tend){
		
			threadData->tstart = (tend - tn2 + 1);
			threadData->tend = tend;
			//ty
			threadData->cut = cut1[1];
		
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
			
			
		}	
		
		
#else
	
		int tsplit = (tstart + tend) >> 1;
		
		if(threadData->id <= tsplit){
		
			threadData->tstart = tstart;
			threadData->tend = tsplit;
			//t
			threadData->cut = cut1[0];
		
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
						
		
		}
		else{
			threadData->tstart = (tsplit + 1);
			threadData->tend = tend;
			//y
			threadData->cut = cut1[1];
		
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
			
		}			
#endif			
	}	
	
	threadData->tstart = tstart;
	threadData->tend = tend;
	threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);	
		
	
	
	
	//t
	cut1[0].dim0[t_index] = t0 + (wt >> 1);
	cut1[0].dim1[t_index] = t1;
	cut1[0].dim0[x_index] = x0;
	cut1[0].dim1[x_index] = x1;
	cut1[0].dim0[y_index] = y0;
	cut1[0].dim1[y_index] = y0 + (wy >> 1);
	cut1[0].dim0[z_index] = z0;
	cut1[0].dim1[z_index] = z0 + (wz >> 1);
	
	//yz
	cut1[1].dim0[t_index] = t0;
	cut1[1].dim1[t_index] = t0 + (wt >> 1);
	cut1[1].dim0[x_index] = x0;
	cut1[1].dim1[x_index] = x1;
	cut1[1].dim0[y_index] = y0 + (wy >> 1);
	cut1[1].dim1[y_index] = y1;
	cut1[1].dim0[z_index] = z0 + (wz >> 1);
	cut1[1].dim1[z_index] = z1;
	
	
	
	if(Ttest(&cut1[0]) || Ttest(&cut1[1])){
#if LB
		
		LB_Calc(depth, cut1, &tn1, &tn2, tn);
		
		if(tn1 != 0 && threadData->id >= tstart && threadData->id <= (tstart+tn1-1)){
		
			threadData->tstart = tstart;
			threadData->tend = tstart+tn1-1;
			//z
			threadData->cut = cut1[0];
			
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
		
		}
		if(tn2 != 0 && threadData->id >= (tend-tn2+1) && threadData->id <= tend){
		
			threadData->tstart = (tend - tn2 + 1);
			threadData->tend = tend;
			//ty
			threadData->cut = cut1[1];
			
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
		}	
		
		
#else		
		int tsplit = (tstart + tend) >> 1;	
		
		if(threadData->id <= tsplit){
		
			threadData->tstart = tstart;
			threadData->tend = tsplit;
			//tz
			threadData->cut = cut1[0];
				
		}
		else{
		
			threadData->tstart = tsplit+1;
			threadData->tend = tend;
			//yz
			threadData->cut = cut1[1];
		}
	
		threadData->depth++;
		PCOTraverser(threadData);
		threadData->depth--;
	
		
#endif			
	}	
	
	threadData->tstart = tstart;
	threadData->tend = tend;
	threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);	
	
	cut1[0].dim0[t_index] = t0 + (wt >> 1);
	cut1[0].dim1[t_index] = t1;
	cut1[0].dim0[x_index] = x0;
	cut1[0].dim1[x_index] = x1;
	cut1[0].dim0[y_index] = y0;
	cut1[0].dim1[y_index] = y0 + (wy >> 1);
	cut1[0].dim0[z_index] = z0 + (wz >> 1);
	cut1[0].dim1[z_index] = z1;
	
	//ty
	cut1[1].dim0[t_index] = t0 + (wt >> 1);
	cut1[1].dim1[t_index] = t1;
	cut1[1].dim0[x_index] = x0;
	cut1[1].dim1[x_index] = x1;
	cut1[1].dim0[y_index] = y0 + (wy >> 1);
	cut1[1].dim1[y_index] = y1;
	cut1[1].dim0[z_index] = z0;
	cut1[1].dim1[z_index] = z0 + (wz >> 1);
	
	
	
	
	
	if(Ttest(&cut1[0]) || Ttest(&cut1[1])){
	
#if LB
		LB_Calc(depth, cut1, &tn1, &tn2, tn);
		
		if(tn1 != 0 && threadData->id >= tstart && threadData->id <= (tstart+tn1-1)){
		
			threadData->tstart = tstart;
			threadData->tend = tstart+tn1-1;
			//t
			threadData->cut = cut1[0];
		
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
			
			
			
		
		}
		if(tn2 != 0 && threadData->id >= (tend-tn2+1) && threadData->id <= tend){
		
			threadData->tstart = (tend - tn2 + 1);
			threadData->tend = tend;
			//y
			threadData->cut = cut1[1];
		
			threadData->depth++;
			PCOTraverser(threadData);
			threadData->depth--;
			
			
			
		}
		

#else
		int tsplit = (tstart + tend) >> 1;
		if(threadData->id <= tsplit){
		
			threadData->tstart = tstart;
			threadData->tend = tsplit;
			//tz
			threadData->cut = cut1[0];		
		
		}
		else{
		
			threadData->tstart = tsplit+1;
			threadData->tend = tend;
			//yz
			threadData->cut = cut1[1];
		}
	
		threadData->depth++;
		PCOTraverser(threadData);
		threadData->depth--;

#endif		
			
	}
	
	threadData->tstart = tstart;
	threadData->tend = tend;
	threadData->barrierTree->sync(tstart, tend, &threadData->threadWaitTime);		

	//tyz
	cut->dim0[t_index] = t0 + (wt >> 1);
	cut->dim1[t_index] = t1;
	cut->dim0[y_index] = y0 + (wy >> 1);
	cut->dim1[y_index] = y1;
	cut->dim0[z_index] = z0 + (wz >> 1);
	cut->dim1[z_index] = z1;
	
	if(Ttest(cut)){
		threadData->depth++;
		PCOTraverser(threadData);
		threadData->depth--;
	}
	
}
#endif
/**/


#if D3D
void LB_PCOTraverser(CO_THREAD_DATA *threadData, int dim = 3, int* counter = 0) {
	
	
	int tstart = threadData->tstart;
	int tend = threadData->tend;
	int tn = tend-tstart+1;
	int depth = threadData->depth;
	
	
	
	
	DOMAIN_3D_CUT *cut = &threadData->cut;
	
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	
	int z0 = cut->dim0[z_index];
	int z1 = cut->dim1[z_index];
	
	int wt = cut->dim1[t_index] - t0;
	int wy = cut->dim1[y_index] - y0;
	int wz = cut->dim1[z_index] - z0;
	
	if (wt <= M_T_BASE || wy <= M_Y_BASE || wz <= M_Z_BASE) {
		//if(t1 < 100 && y1 < 100 && z1 < 100)
			//cout<<"here2"<<endl;
		//cout<<t0<<" "<<t1<<" "<<y0<<" "<<y1<<" "<<z0<<" "<<z1<<endl;	
		
		LB_COComputeKernel(threadData);
		return;
	}
	
	
	DOMAIN_3D_CUT cut1[8];
	
	//000
	cut1[0].dim0[t_index] = t0;
	cut1[0].dim1[t_index] = t0 + (wt >> 1);
	cut1[0].dim0[x_index] = x0;
	cut1[0].dim1[x_index] = x1;
	cut1[0].dim0[y_index] = y0;
	cut1[0].dim1[y_index] = y0 + (wy >> 1);
	cut1[0].dim0[z_index] = z0;
	cut1[0].dim1[z_index] = z0 + (wz >> 1);

	//t
	cut1[1].dim0[t_index] = t0 + (wt >> 1);
	cut1[1].dim1[t_index] = t1;
	cut1[1].dim0[x_index] = x0;
	cut1[1].dim1[x_index] = x1;
	cut1[1].dim0[y_index] = y0;
	cut1[1].dim1[y_index] = y0 + (wy >> 1);
	cut1[1].dim0[z_index] = z0;
	cut1[1].dim1[z_index] = z0 + (wz >> 1);
	
	//y
	cut1[2].dim0[t_index] = t0;
	cut1[2].dim1[t_index] = t0 + (wt >> 1);
	cut1[2].dim0[x_index] = x0;
	cut1[2].dim1[x_index] = x1;
	cut1[2].dim0[y_index] = y0 + (wy >> 1);
	cut1[2].dim1[y_index] = y1;
	cut1[2].dim0[z_index] = z0;
	cut1[2].dim1[z_index] = z0 + (wz >> 1);
	
	//z
	cut1[3].dim0[t_index] = t0;
	cut1[3].dim1[t_index] = t0 + (wt >> 1);
	cut1[3].dim0[x_index] = x0;
	cut1[3].dim1[x_index] = x1;
	cut1[3].dim0[y_index] = y0;
	cut1[3].dim1[y_index] = y0 + (wy >> 1);
	cut1[3].dim0[z_index] = z0 + (wz >> 1);
	cut1[3].dim1[z_index] = z1;
		
	
	//ty
	cut1[4].dim0[t_index] = t0 + (wt >> 1);
	cut1[4].dim1[t_index] = t1;
	cut1[4].dim0[x_index] = x0;
	cut1[4].dim1[x_index] = x1;
	cut1[4].dim0[y_index] = y0 + (wy >> 1);
	cut1[4].dim1[y_index] = y1;
	cut1[4].dim0[z_index] = z0;
	cut1[4].dim1[z_index] = z0 + (wz >> 1);	
	
	
	//tz
	cut1[5].dim0[t_index] = t0 + (wt >> 1);
	cut1[5].dim1[t_index] = t1;
	cut1[5].dim0[x_index] = x0;
	cut1[5].dim1[x_index] = x1;
	cut1[5].dim0[y_index] = y0;
	cut1[5].dim1[y_index] = y0 + (wy >> 1);
	cut1[5].dim0[z_index] = z0 + (wz >> 1);
	cut1[5].dim1[z_index] = z1;
	
	
	//yz
	cut1[6].dim0[t_index] = t0;
	cut1[6].dim1[t_index] = t0 + (wt >> 1);
	cut1[6].dim0[x_index] = x0;
	cut1[6].dim1[x_index] = x1;
	cut1[6].dim0[y_index] = y0 + (wy >> 1);
	cut1[6].dim1[y_index] = y1;
	cut1[6].dim0[z_index] = z0 + (wz >> 1);
	cut1[6].dim1[z_index] = z1;
	
	
	//tyz
	cut1[7].dim0[t_index] = t0 + (wt >> 1);
	cut1[7].dim1[t_index] = t1;
	cut1[7].dim0[x_index] = x0;
	cut1[7].dim1[x_index] = x1;
	cut1[7].dim0[y_index] = y0 + (wy >> 1);
	cut1[7].dim1[y_index] = y1;
	cut1[7].dim0[z_index] = z0 + (wz >> 1);
	cut1[7].dim1[z_index] = z1;
	
	threadData->tstart = threadData->id;
	threadData->tend = threadData->id;
	int id = threadData->id;
	
	for(int i = 0; i < 8/tn; i++){
		//for(int j = 0; j < tn; j++){
			int k = id - tstart;
			/*if(tn*i+id >= 2 && id == 0){
				cout<<"here is a problem "<<tn*i+id<<" "<<tn<<" "<<i<<" "<<id<<endl;
				char tt;
				cin>>tt;
			}*/
			threadData->cut = cut1[tn*i+k];
			if(Ttest(&threadData->cut)){
				threadData->depth++;
				LB_PCOTraverser(threadData);
				threadData->depth--;
				/*if(threadData->depth > 0){
					int div = pow(2, threadData->depth);
					int tdiv_size = T/div;
					int ydiv_size = Y/div;
					int zdiv_size = Z/div;
					int y = y1/ydiv_size - 1;
					int z = z1/zdiv_size - 1;
					int t = t1/tdiv_size - 1;
					int index = get(threadData->depth, y, z, t);
					int counter = lb_array[get(threadData->depth+1, 2*y, 2*z, 2*t)] + lb_array[get(threadData->depth+1, 2*y+1, 2*z, 2*t)] + lb_array[get(threadData->depth+1, 2*y, 2*z+1, 2*t)] + lb_array[get(threadData->depth+1, 2*y+1, 2*z+1, 2*t)] + lb_array[get(threadData->depth+1, 2*y, 2*z, 2*t+1)] + lb_array[get(threadData->depth+1, 2*y+1, 2*z, 2*t+1)] + lb_array[get(threadData->depth+1, 2*y, 2*z+1, 2*t+1)] + lb_array[get(threadData->depth+1, 2*y+1, 2*z+1, 2*t+1)]; 
					lb_array[index] = counter;
				}*/
			}
		//}
	}
	
	/*if(depth == 0){
		threadData->tstart = 0;
		threadData->tend = NUM_THREADS-1;
		threadData->barrierTree->sync(tstart, tend, 0);		
		if(threadData->id == 0){
		
			CollectAll();
		
			int sum = 0;
			for(int i = 0; i < 8; i++)
				sum += lb_array[i]; 
	
			//cout<<"Final "<<sum<<endl;
			//cout<<lb_array[1]<<" "<<lb_array[4]<<endl;
		}
		threadData->threadWaitTime = 0.0;	
		
		threadData->cut.dim0[t_index] = t0;
		threadData->cut.dim1[t_index] = t1;
		threadData->cut.dim0[x_index] = x0;
		threadData->cut.dim1[x_index] = x1;
		threadData->cut.dim0[y_index] = y0;
		threadData->cut.dim1[y_index] = y1;
		threadData->cut.dim0[z_index] = z0;
		threadData->cut.dim1[z_index] = z1;
		
		//printf("%08x\n", threadData->magic);
		//fflush(stdout);
		//cout<<t0<<" "<<t1<<" "<<x0<<" "<<x1<<" "<<y0<<" "<<y1<<" "<<z0<<" "<<z1<<endl;
		threadData->barrierTree->sync(tstart, tend, 0);		
		
	}*/		
}
#else
//here is the function to be investigated
void LB_PCOTraverser(CO_THREAD_DATA *threadData, int& counter) {
	
	
	
	int tstart = threadData->tstart;
	int tend = threadData->tend;
	int tn = tend-tstart+1;
	int depth = threadData->depth;
	
	int id = threadData->id;
	int k = id - tstart;
	float tn1, tn2;
	
	int i = 0; 
	int j = 0;
	int accum = 0; 
	int index = 0;
	int caseNr= 0;
	int ind1 = 0;
	int ind2 = 0;
	int ind3 = 0;
	int ind4 = 0;
	int pind[DIM] = {0};
	int maxSize = 1;
	
	
	
	DOMAIN_3D_CUT cut1[16];
	
	int pairs4D[]= {0b0000, 0b0001, 0b0010, 0b0100, 0b1000, 0b0101, 0b0110, 0b0011, 0b1001, 0b1100, 0b1010, 0b1101, 0b1011, 0b0111, 0b1110, 0b1111}; 
	int pairs3D[]= {0b000, 0b001, 0b010, 0b100, 0b011, 0b110, 0b101, 0b111}; 
	int pairs2D[] = {0b00, 0b01, 0b10, 0b11};
 	
	
		
	int dim0[DIM];
	int dim1[DIM];
	int wdim[DIM];
	
	
	

	//int ind1 = t_index, ind2 = x_index, ind3 = y_index, ind4 = z_index;
	
	DOMAIN_3D_CUT *cut = &threadData->cut;
	
	dim0[t_index] = cut->dim0[t_index];
	dim1[t_index] = cut->dim1[t_index];
	
	dim0[x_index] = cut->dim0[x_index];
	dim1[x_index] = cut->dim1[x_index];
	
	dim0[y_index] = cut->dim0[y_index];
	dim1[y_index] = cut->dim1[y_index];
	
	dim0[z_index] = cut->dim0[z_index];
	dim1[z_index] = cut->dim1[z_index];
	
	wdim[t_index] = cut->dim1[t_index] - dim0[t_index];
	wdim[x_index] = cut->dim1[x_index] - dim0[x_index];
	wdim[y_index] = cut->dim1[y_index] - dim0[y_index];
	wdim[z_index] = cut->dim1[z_index] - dim0[z_index];
	
	
	int indices[4] = {0};
	
	//if(xdiv_size[depth] > 1)
		indices[x_index] =  dim0[x_index]/xdiv_size[depth];
	//if(ydiv_size[depth] > 1)
		indices[y_index] =  dim0[y_index]/ydiv_size[depth];
	//if(zdiv_size[depth] > 1)
		indices[z_index] =  dim0[z_index]/zdiv_size[depth];
	//if(tdiv_size[depth] > 1)
		indices[t_index] =  dim0[t_index]/tdiv_size[depth];
	
	index = get(depth, indices[x_index], indices[y_index], indices[z_index], indices[t_index]);

	


	// input values: current tileSize and parallel baseSize
	int relSize[]= {
   		wdim[t_index]/M_BASE[t_index],
   		wdim[x_index]/M_BASE[x_index],
	   	wdim[y_index]/M_BASE[y_index],
	   	wdim[z_index]/M_BASE[z_index],
	};
	
	
	
	// choose scheduler
	switch( CURRENT_SCHEDULER ) {
		
		case SCHEDULER_PARALLELISM:
		   
		   for(i = 0; i < DIM; i++) {
		     if(relSize[i] > 1) {
		       pind[caseNr] = i; 
		       caseNr++;
		     }
		   }
		   j = caseNr;
		   for(i = 0; i < DIM; i++){
		   	if(relSize[i] <= 1){
		     	pind[j++] = i; 
		     }
		   }
		   break;
		case SCHEDULER_FORM:
		   
		   for(i = 0; i < DIM; i++) {
		     if(maxSize<relSize[i]) 
		     	maxSize = relSize[i];
		   }
		   
		   for(i = 0; i < DIM; i++) {
		   			   	 
		   	if(relSize[i] == maxSize) {
			
		       pind[caseNr] = i;
		       caseNr++;
		    }
		   }
		   j = caseNr;
		   for(i = 0; i < DIM; i++){
		   	if(relSize[i] != maxSize){
		     	pind[j++] = i; 
		     }
		   }
		   if(maxSize == 1) 
		   	caseNr= 0;
		   break;
	}
	ind1 = pind[0];
	ind2 = pind[1];
	ind3 = pind[2];
	ind4 = pind[3];
	
	
 	switch(caseNr){
 		
 		case 4:
 			
 			threadData->tstart = threadData->id;
			threadData->tend = threadData->id;
			
			for(i = 0; i < 16/tn; i++){
	
				
				
				cut1[(tn*i+k)].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs4D[(tn*i+k)]&(1<<0))));
				cut1[(tn*i+k)].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((15-pairs4D[(tn*i+k)])&(1<<0))));
				cut1[(tn*i+k)].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs4D[(tn*i+k)]&(1<<1))));
				cut1[(tn*i+k)].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((15-pairs4D[(tn*i+k)])&(1<<1))));
				cut1[(tn*i+k)].dim0[ind3] = dim0[ind3] + ((wdim[ind3] >> 1) * bool((pairs4D[(tn*i+k)]&(1<<2))));
				cut1[(tn*i+k)].dim1[ind3] = dim1[ind3] - ((wdim[ind3] >> 1) * bool(((15-pairs4D[(tn*i+k)])&(1<<2))));
				cut1[(tn*i+k)].dim0[ind4] = dim0[ind4] + ((wdim[ind4] >> 1) * bool((pairs4D[(tn*i+k)]&(1<<3))));
				cut1[(tn*i+k)].dim1[ind4] = dim1[ind4] - ((wdim[ind4] >> 1) * bool(((15-pairs4D[(tn*i+k)])&(1<<3))));
	
	
							
				threadData->cut = cut1[tn*i+k];
				
				if(Ttest(&threadData->cut)){
					
					threadData->depth++;
					LB_PCOTraverser(threadData, counter);
					threadData->depth--;
					
					if(depth > 0){
						
						
						lb_array4D[index] += counter;//accumulate into the parent node
					}
					
					accum += counter;
					
				}
													
			}
			counter = accum;
			
			
		break;
		
		case 3:
			
			threadData->tstart = threadData->id;
			threadData->tend = threadData->id;
			
			for(i = 0; i < 8/tn; i++){
	
				cut1[(tn*i+k)].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs3D[(tn*i+k)]&(1<<0))));
				cut1[(tn*i+k)].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((7-pairs3D[(tn*i+k)])&(1<<0))));
				cut1[(tn*i+k)].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs3D[(tn*i+k)]&(1<<1))));
				cut1[(tn*i+k)].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((7-pairs3D[(tn*i+k)])&(1<<1))));
				cut1[(tn*i+k)].dim0[ind3] = dim0[ind3] + ((wdim[ind3] >> 1) * bool((pairs3D[(tn*i+k)]&(1<<2))));
				cut1[(tn*i+k)].dim1[ind3] = dim1[ind3] - ((wdim[ind3] >> 1) * bool(((7-pairs3D[(tn*i+k)])&(1<<2))));
				cut1[(tn*i+k)].dim0[ind4] = dim0[ind4];
				cut1[(tn*i+k)].dim1[ind4] = dim1[ind4];
	
	
							
				threadData->cut = cut1[tn*i+k];
				
				if(Ttest(&threadData->cut)){
					
					threadData->depth++;
					
					LB_PCOTraverser(threadData, counter); 
					
					threadData->depth--;
					
					if(depth > 0){
						
						lb_array4D[index] += counter;
						accum += counter;
					}						
				}						
			}
			counter = accum;
			
				
			break;
		
		case 2:
 			
			threadData->tstart = threadData->id;
			threadData->tend = threadData->id;
			
			for(i = 0; i < 4/tn; i++){
	
				
				
				cut1[(tn*i+k)].dim0[ind1] = dim0[ind1] + ((wdim[ind1] >> 1) * bool((pairs2D[(tn*i+k)]&(1<<0))));
				cut1[(tn*i+k)].dim1[ind1] = dim1[ind1] - ((wdim[ind1] >> 1) * bool(((3-pairs2D[(tn*i+k)])&(1<<0))));
				cut1[(tn*i+k)].dim0[ind2] = dim0[ind2] + ((wdim[ind2] >> 1) * bool((pairs2D[(tn*i+k)]&(1<<1))));
				cut1[(tn*i+k)].dim1[ind2] = dim1[ind2] - ((wdim[ind2] >> 1) * bool(((3-pairs2D[(tn*i+k)])&(1<<1))));
				cut1[(tn*i+k)].dim0[ind3] = dim0[ind3];
				cut1[(tn*i+k)].dim1[ind3] = dim1[ind3];
				cut1[(tn*i+k)].dim0[ind4] = dim0[ind4];
				cut1[(tn*i+k)].dim1[ind4] = dim1[ind4];
	
	
							
				threadData->cut = cut1[tn*i+k];
				
				if(Ttest(&threadData->cut)){
					
					threadData->depth++;
					LB_PCOTraverser(threadData, counter);
					threadData->depth--;
	
					if(depth > 0){
					
						lb_array4D[index] += counter;//accumulate into the parent node
					}
					accum += counter;
					
				}									
			}
			counter = accum;
			
			
			break;
		
		case 1:
 			 	 			 						 			 	
 			 	cut1[0].dim0[ind1] = dim0[ind1];
				cut1[0].dim1[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
				cut1[0].dim0[ind2] = dim0[ind2];
				cut1[0].dim1[ind2] = dim1[ind2];
				cut1[0].dim0[ind3] = dim0[ind3];
				cut1[0].dim1[ind3] = dim1[ind3];
				cut1[0].dim0[ind4] = dim0[ind4];
				cut1[0].dim1[ind4] = dim1[ind4];
		
								
				cut1[1].dim0[ind1] = dim0[ind1] + (wdim[ind1] >> 1);
				cut1[1].dim1[ind1] = dim1[ind1];
				cut1[1].dim0[ind2] = dim0[ind2];
				cut1[1].dim1[ind2] = dim1[ind2];
				cut1[1].dim0[ind3] = dim0[ind3];
				cut1[1].dim1[ind3] = dim1[ind3];
				cut1[1].dim0[ind4] = dim0[ind4];
				cut1[1].dim1[ind4] = dim1[ind4];
				
				
				tn1 = ceilf((float)tn/2.0f);
				tn2 = tn - tn1;
				if(tn2 == 0)
					tn2 = 1.0f;
				
					
				if(Ttest(&cut1[0]) && id >= tstart && threadData->id <= (tstart+tn1-1)){
					
					threadData->tstart = tstart;
					threadData->tend = (tstart+tn1-1);
					threadData->cut = cut1[0];
					
					threadData->depth++;
					LB_PCOTraverser(threadData, counter);
					threadData->depth--;
					
					if(id == tstart && depth > 0){
					
					
						lb_array4D[index] += counter;
						accum += counter;
					}

					
					
				}
				if(Ttest(&cut1[1]) && id >= (tend-tn2+1) && threadData->id <= tend){
	 				
	 				threadData->tstart = (tend-tn2+1);
					threadData->tend = tend;
					threadData->cut = cut1[1];
					
					threadData->depth++;
					LB_PCOTraverser(threadData, counter);
					threadData->depth--;
					
					if(id == tend-tn2+1 && depth > 0){
					
					
						lb_array4D[index] += counter;
						accum += counter;
					}
					
				}
				counter = accum;
			break;
		
		case 0:
			if(id == tstart)
				LB_COComputeKernel(threadData, &counter);
			break;
		
		
 	}
 	
 	
	
}
//dedicated 4D version
/*
void LB_PCOTraverser(CO_THREAD_DATA *threadData, int dim = 4) {
	
	
	int tstart = threadData->tstart;
	int tend = threadData->tend;
	int tn = tend-tstart+1;
	int depth = threadData->depth;
	
	
	
	
	DOMAIN_3D_CUT *cut = &threadData->cut;
	
	
	int t0 = cut->dim0[t_index];
	int t1 = cut->dim1[t_index];
	
	int x0 = cut->dim0[x_index];
	int x1 = cut->dim1[x_index];
	
	int y0 = cut->dim0[y_index];
	int y1 = cut->dim1[y_index];
	
	int z0 = cut->dim0[z_index];
	int z1 = cut->dim1[z_index];
	
	int wt = cut->dim1[t_index] - t0;
	int wx = cut->dim1[x_index] - x0;
	int wy = cut->dim1[y_index] - y0;
	int wz = cut->dim1[z_index] - z0;
	
	if (wt <= M_T_BASE || wx <= M_X_BASE || wy <= M_Y_BASE || wz <= M_Z_BASE) {
				
		LB_COComputeKernel(threadData);
		return;
	}
	
	
	DOMAIN_3D_CUT cut1[16];
	
	//0000
	cut1[0].dim0[t_index] = t0;
	cut1[0].dim1[t_index] = t0 + (wt >> 1);
	cut1[0].dim0[x_index] = x0;
	cut1[0].dim1[x_index] = x0 + (wx >> 1);
	cut1[0].dim0[y_index] = y0;
	cut1[0].dim1[y_index] = y0 + (wy >> 1);
	cut1[0].dim0[z_index] = z0;
	cut1[0].dim1[z_index] = z0 + (wz >> 1);

	//t
	cut1[1].dim0[t_index] = t0 + (wt >> 1);
	cut1[1].dim1[t_index] = t1;
	cut1[1].dim0[x_index] = x0;
	cut1[1].dim1[x_index] = x0 + (wx >> 1);
	cut1[1].dim0[y_index] = y0;
	cut1[1].dim1[y_index] = y0 + (wy >> 1);
	cut1[1].dim0[z_index] = z0;
	cut1[1].dim1[z_index] = z0 + (wz >> 1);
	
	//x
	cut1[2].dim0[t_index] = t0;
	cut1[2].dim1[t_index] = t0 + (wt >> 1);
	cut1[2].dim0[x_index] = x0 + (wx >> 1);
	cut1[2].dim1[x_index] = x1;
	cut1[2].dim0[y_index] = y0;
	cut1[2].dim1[y_index] = y0 + (wy >> 1);
	cut1[2].dim0[z_index] = z0;
	cut1[2].dim1[z_index] = z0 + (wz >> 1);
	
	//z
	cut1[3].dim0[t_index] = t0;
	cut1[3].dim1[t_index] = t0 + (wt >> 1);
	cut1[3].dim0[x_index] = x0;
	cut1[3].dim1[x_index] = x0 + (wx >> 1);
	cut1[3].dim0[y_index] = y0;
	cut1[3].dim1[y_index] = y0 + (wy >> 1);
	cut1[3].dim0[z_index] = z0 + (wz >> 1);
	cut1[3].dim1[z_index] = z1;
	
	//y
	cut1[4].dim0[t_index] = t0;
	cut1[4].dim1[t_index] = t0 + (wt >> 1);
	cut1[4].dim0[x_index] = x0;
	cut1[4].dim1[x_index] = x0 + (wx >> 1);
	cut1[4].dim0[y_index] = y0 + (wy >> 1);
	cut1[4].dim1[y_index] = y1;
	cut1[4].dim0[z_index] = z0;
	cut1[4].dim1[z_index] = z0 + (wz >> 1);
		
	//tz
	cut1[5].dim0[t_index] = t0 + (wt >> 1);
	cut1[5].dim1[t_index] = t1;
	cut1[5].dim0[x_index] = x0;
	cut1[5].dim1[x_index] = x0 + (wx >> 1);
	cut1[5].dim0[y_index] = y0;
	cut1[5].dim1[y_index] = y0 + (wy >> 1);
	cut1[5].dim0[z_index] = z0 + (wz >> 1);
	cut1[5].dim1[z_index] = z1;
	
	
	//tx
	cut1[6].dim0[t_index] = t0 + (wt >> 1);
	cut1[6].dim1[t_index] = t1;
	cut1[6].dim0[x_index] = x0 + (wx >> 1);
	cut1[6].dim1[x_index] = x1;
	cut1[6].dim0[y_index] = y0;
	cut1[6].dim1[y_index] = y0 + (wy >> 1);
	cut1[6].dim0[z_index] = z0;
	cut1[6].dim1[z_index] = z0 + (wz >> 1);
	
	
	//ty
	cut1[7].dim0[t_index] = t0 + (wt >> 1);
	cut1[7].dim1[t_index] = t1;
	cut1[7].dim0[x_index] = x0;
	cut1[7].dim1[x_index] = x0 + (wx >> 1);
	cut1[7].dim0[y_index] = y0 + (wy >> 1);
	cut1[7].dim1[y_index] = y1;
	cut1[7].dim0[z_index] = z0;
	cut1[7].dim1[z_index] = z0 + (wz >> 1);
	
	
	//xy
	cut1[8].dim0[t_index] = t0;
	cut1[8].dim1[t_index] = t0 + (wt >> 1);
	cut1[8].dim0[x_index] = x0 + (wx >> 1);
	cut1[8].dim1[x_index] = x1;
	cut1[8].dim0[y_index] = y0 + (wy >> 1);
	cut1[8].dim1[y_index] = y1;
	cut1[8].dim0[z_index] = z0;
	cut1[8].dim1[z_index] = z0 + (wz >> 1);
	
	//yz
	cut1[9].dim0[t_index] = t0;
	cut1[9].dim1[t_index] = t0 + (wt >> 1);
	cut1[9].dim0[x_index] = x0;
	cut1[9].dim1[x_index] = x0 + (wx >> 1);
	cut1[9].dim0[y_index] = y0 + (wy >> 1);
	cut1[9].dim1[y_index] = y1;
	cut1[9].dim0[z_index] = z0 + (wz >> 1);
	cut1[9].dim1[z_index] = z1;
	
	//xz
	cut1[10].dim0[t_index] = t0;
	cut1[10].dim1[t_index] = t0 + (wt >> 1);
	cut1[10].dim0[x_index] = x0 + (wx >> 1);
	cut1[10].dim1[x_index] = x1;
	cut1[10].dim0[y_index] = y0;
	cut1[10].dim1[y_index] = y0 + (wy >> 1);
	cut1[10].dim0[z_index] = z0 + (wz >> 1);
	cut1[10].dim1[z_index] = z1;
	
	//tyz
	cut1[11].dim0[t_index] = t0 + (wt >> 1);
	cut1[11].dim1[t_index] = t1;
	cut1[11].dim0[x_index] = x0;
	cut1[11].dim1[x_index] = x0 + (wx >> 1);
	cut1[11].dim0[y_index] = y0 + (wy >> 1);
	cut1[11].dim1[y_index] = y1;
	cut1[11].dim0[z_index] = z0 + (wz >> 1);
	cut1[11].dim1[z_index] = z1;
	
	//xzt
	cut1[12].dim0[t_index] = t0 + (wt >> 1);
	cut1[12].dim1[t_index] = t1;
	cut1[12].dim0[x_index] = x0 + (wx >> 1);
	cut1[12].dim1[x_index] = x1;
	cut1[12].dim0[y_index] = y0;
	cut1[12].dim1[y_index] = y0 + (wy >> 1);
	cut1[12].dim0[z_index] = z0 + (wz >> 1);
	cut1[12].dim1[z_index] = z1;
	
	//xyt
	cut1[13].dim0[t_index] = t0 + (wt >> 1);
	cut1[13].dim1[t_index] = t1;
	cut1[13].dim0[x_index] = x0 + (wx >> 1);
	cut1[13].dim1[x_index] = x1;
	cut1[13].dim0[y_index] = y0 + (wy >> 1);
	cut1[13].dim1[y_index] = y1;
	cut1[13].dim0[z_index] = z0;
	cut1[13].dim1[z_index] = z0 + (wz >> 1);
	
	//xyz
	cut1[14].dim0[t_index] = t0;
	cut1[14].dim1[t_index] = t0 + (wt >> 1);
	cut1[14].dim0[x_index] = x0 + (wx >> 1);
	cut1[14].dim1[x_index] = x1;
	cut1[14].dim0[y_index] = y0 + (wy >> 1);
	cut1[14].dim1[y_index] = y1;
	cut1[14].dim0[z_index] = z0 + (wz >> 1);
	cut1[14].dim1[z_index] = z1;
	
	//txyz
	cut1[15].dim0[t_index] = t0 + (wt >> 1);
	cut1[15].dim1[t_index] = t1;
	cut1[15].dim0[x_index] = x0 + (wx >> 1);
	cut1[15].dim1[x_index] = x1;
	cut1[15].dim0[y_index] = y0 + (wy >> 1);
	cut1[15].dim1[y_index] = y1;
	cut1[15].dim0[z_index] = z0 + (wz >> 1);
	cut1[15].dim1[z_index] = z1;
	
	threadData->tstart = threadData->id;
	threadData->tend = threadData->id;
	int id = threadData->id;
	
	for(int i = 0; i < 16/tn; i++){
		
			int k = id - tstart;
			
			threadData->cut = cut1[tn*i+k];
			if(Ttest(&threadData->cut)){
				threadData->depth++;
				LB_PCOTraverser(threadData);
				threadData->depth--;
				
			}
		
	}
}/***/
#endif
void *PcoThread(void *ptr) {
	
	CO_THREAD_DATA *threadData = (CO_THREAD_DATA *)ptr;
	DOMAIN_3D_CUT save_cut = threadData->cut;

	int tstart = threadData->tstart;
	int tend = threadData->tend;
	SetupCurrentThread((threadData->id)/((NUM_THREADS/4)>1?(NUM_THREADS/4):1));
	if(tstart != tend && threadData->id <= min(NUM_THREADS-1, 7)){
		threadData->tend = min(NUM_THREADS-1, 7);
		int counter = 0;
		LB_PCOTraverser(threadData, counter);
		threadData->tstart = 0;
		threadData->tend = tend;
		threadData->barrierTree->sync(0, min(NUM_THREADS-1, 7), 0);
		
#if D3D
		if(threadData->id == 0){
			CollectAll();			
		}	
#endif		
#if CHECK_LB		
		if(threadData->id == 0){
			CheckAll();	
			
			
		}	
#endif		
		threadData->threadWaitTime = 0.0;	
		threadData->tstart = 0;
		threadData->tend = NUM_THREADS-1;
		threadData->cut = save_cut;
		
	}
	threadData->barrierTree->sync(0, NUM_THREADS-1, 0);	
	PCOTraverser(threadData);
	
	
	return 0;
}

int main(int argc, char**argv) {
#ifndef _WIN32
	sched_param sparam;
	sparam.sched_priority = sched_get_priority_max(SCHED_FIFO) - 2;
	int ret = sched_setscheduler(getpid(), SCHED_FIFO, &sparam);
	if (ret < 0) {
		printf("FIFO Priority change failed! %d\n", ret);
	}
#endif
	double tsample[STAT_SAMPLES];
	PerfTimer *timer = new PerfTimer();
	/*
	int bsize = (int) floor((double)CACHE_SIZE / (2.8 * sizeof(float) * SIZE_X * SIZE_Y));
	if (bsize == 0) {
		bsize = 1;
	} else if (bsize > TIME_STEPS) {
		bsize = TIME_STEPS;
	}

	printf("=== SINGLE PRECISION ===\n");
	printf("ATS1 Block size: %i\n", bsize);

	int ats2bsize = 0;
	int ats2sd = 0;
	CalcATS2(4, ats2bsize, ats2sd, SIZE_X, SIZE_Y, SIZE_Z);
	printf("ATS2 Block size: %i Padding offset: %i\n", ats2bsize, ats2sd);
	
	
	// DP
	bsize = (int) floor((double)CACHE_SIZE / (2.8 * sizeof(double) * SIZE_X * SIZE_Y));
	if (bsize == 0) {
		bsize = 1;
	} else if (bsize > TIME_STEPS) {
		bsize = TIME_STEPS;
	}

	printf("\n=== DOUBLE PRECISION ===\n");
	printf("ATS1 Block size: %i\n", bsize);

	CalcATS2(8, ats2bsize, ats2sd, SIZE_X, SIZE_Y, SIZE_Z);
	printf("ATS2 Block size: %i Padding offset: %i\n", ats2bsize, ats2sd);
	*/
	double *dpdata1 = (double *)_mm_malloc(DATA_SIZE * sizeof(double) * 2, 16);
	InitDataDP(dpdata1);

	/*double *dpdata2 = (double *)_mm_malloc(DATA_SIZE * sizeof(double) * 2, 16);
	InitDataDP(dpdata2);
	
	double *dpdata3 = (double *)_mm_malloc(DATA_SIZE * sizeof(double) * 2, 16);
	InitDataDP(dpdata3);
	*/
	double *dpdata4 = (double *)_mm_malloc(DATA_SIZE * sizeof(double) * 2, 16);
	InitDataDP(dpdata4);
//#undef NUM_THREADS
//#define NUM_THREADS 1	
#if NAIVE	
	for (int i = 0; i < STAT_SAMPLES; i++) {
		
		FlushCacheMemory(dpdata1, DATA_SIZE * sizeof(double));
		timer->start();
#if _3D		
		Naive(NaiveDP, dpdata1, SIZE_Y);
#elif _2D
		Naive(NaiveDP, dpdata1, SIZE_X);
#else
		Naive(NaiveDP, dpdata1, SIZE_Z);
#endif					
		tsample[i] = timer->elapsed();
//		printf("time: %f\n", tsample[i]);
	}
	
	
#endif
/*
	for (int i = 0; i < STAT_SAMPLES; i++) {
		FlushCacheMemory(dpdata2, DATA_SIZE * sizeof(double));
		timer->start();
		ATS1(ATS1DP, dpdata2, bsize, SIZE_Z);
		tsample[i] = timer->elapsed();
//		printf("time: %f\n", tsample[i]);
	}

	printf("ATS1 DP average time: %f\n", GetStatTime(tsample, STAT_SAMPLES));
	
	for (int i = 0; i < STAT_SAMPLES; i++) {
		FlushCacheMemory(dpdata3, DATA_SIZE * sizeof(double));
		timer->start();
		ATS2(ATS2DP, dpdata3, ats2bsize, ats2sd, SIZE_Y);
		tsample[i] = timer->elapsed();
//		printf("time: %f\n", tsample[i]);
	}

	printf("ATS2 DP average time: %f\n", GetStatTime(tsample, STAT_SAMPLES));
*/	
	
	
#if _2D
	int sz = 64;
	int factor = 2;
#elif _3D
	int sz = 64;
	int factor = 2;
#else		
	int sz = 512;
	int factor = 10;
#endif	
	int tmp = (sz*1024)/(sizeof(double) * 2);
#if _2D
	M_X_BASE = (tmp/factor); 
#elif _3D
	M_Y_BASE = sqrt(tmp/factor); 
#else	
	M_Y_BASE = cbrt(tmp/factor); 
#endif	
	
#ifndef _2D	
	M_X_BASE = factor*M_Y_BASE;
	M_Y_BASE = pow(2, ceil((double)log((double)M_Y_BASE)/log(2.0)));
	M_Z_BASE = M_Y_BASE;
	S_Z_BASE = 8;
	S_Y_BASE = 8;
#endif	
	if(M_X_BASE > TIME_STEPS+SIZE_X)
		M_X_BASE = SIZE_X;
	M_X_BASE = pow(2, ceil((double)log((double)M_X_BASE)/log(2.0)));
	M_T_BASE = M_X_BASE;
	
	
	if(M_T_BASE > TIME_STEPS)
		M_T_BASE = TIME_STEPS;
		
	S_X_BASE = M_X_BASE;
	S_T_BASE = M_T_BASE;
	
	
	
	cout<<"M_X_BASE = "<<M_X_BASE<<endl<<"M_Y_BASE = "<<M_Y_BASE<<endl<<"M_Z_BASE = "<<M_Z_BASE<<endl<<"M_T_BASE = "<<M_T_BASE<<endl<<endl;
	cout<<"S_X_BASE = "<<S_X_BASE<<endl<<"S_Y_BASE = "<<S_Y_BASE<<endl<<"S_Z_BASE = "<<S_Z_BASE<<endl<<"S_T_BASE = "<<S_T_BASE<<endl;
	BarrierTree *barrierTree = new BarrierTree(50);

	DOMAIN_3D_CUT cut;
	
	cut.dim0[t_index] = 0;
	cut.dim1[t_index] = ((int)pow(2.0, ceil((log(((double)TIME_STEPS)/M_T_BASE) / log(2.0))))) * M_T_BASE;
	cut.dim0[x_index] = 0;
	cut.dim1[x_index] = ((int)pow(2.0, ceil(log(((double)SIZE_X + TIME_STEPS)/M_X_BASE) / log(2.0)))) * M_X_BASE;
	cut.dim0[y_index] = 0;
	cut.dim1[y_index] = ((int)pow(2.0, ceil(log(((double)SIZE_Y + TIME_STEPS)/M_Y_BASE) / log(2.0)))) * M_Y_BASE;
	cut.dim0[z_index] = 0;
	cut.dim1[z_index] = ((int)pow(2.0, ceil(log(((double)SIZE_Z + TIME_STEPS)/M_Z_BASE) / log(2.0)))) * M_Z_BASE;
	if(SIZE_Z == 1){
		cut.dim1[z_index] = 1;
	}
	if(SIZE_Y == 1){
		cut.dim1[y_index] = 1;
	}
	
	
	cout<<cut.dim1[t_index]<<" "<<cut.dim1[x_index]<<" "<<cut.dim1[y_index]<<" "<<cut.dim1[z_index]<<endl;
	
#if D3D	
	MIN_CUT = min((int)(log((cut.dim1[t_index]-cut.dim0[t_index])/M_T_BASE)/log(2)), min((int)(log((cut.dim1[y_index]-cut.dim0[y_index])/M_Y_BASE)/log(2)),(int)(log((cut.dim1[z_index]-cut.dim0[z_index])/M_Z_BASE)/log(2))));
	cout<<MIN_CUT<<endl;
#else	
	
	int max_x = (log((cut.dim1[x_index]-cut.dim0[x_index])/M_X_BASE)/log(2));
	int max_y = (log((cut.dim1[y_index]-cut.dim0[y_index])/M_Y_BASE)/log(2));
	int max_z = (log((cut.dim1[z_index]-cut.dim0[z_index])/M_Z_BASE)/log(2));
	int max_t = (log((cut.dim1[t_index]-cut.dim0[t_index])/M_T_BASE)/log(2));
	
	MIN_CUT4D = max(max_x, max(max_t, max(max_y, max_z)));
	cout<<"Max Depth "<<MIN_CUT4D<<endl;

	tdiv_size = new int[MIN_CUT4D+1];
	xdiv_size = new int[MIN_CUT4D+1];
	ydiv_size = new int[MIN_CUT4D+1];
	zdiv_size = new int[MIN_CUT4D+1];
	xarr_size = new int[MIN_CUT4D+1];
	yarr_size = new int[MIN_CUT4D+1];
	zarr_size = new int[MIN_CUT4D+1];
	tarr_size = new int[MIN_CUT4D+1];
	offset    = new int[MIN_CUT4D+1]; 

	tdiv_size[0] = cut.dim1[t_index]-cut.dim0[t_index];
	xdiv_size[0] = cut.dim1[x_index]-cut.dim0[x_index];
	ydiv_size[0] = cut.dim1[y_index]-cut.dim0[y_index];
	zdiv_size[0] = cut.dim1[z_index]-cut.dim0[z_index];
	
	tarr_size[0] = 1;
	xarr_size[0] = 1;
	yarr_size[0] = 1;
	zarr_size[0] = 1;
	
	int counter = 0;
	offset[1] = 0;
	int x = 1;
	for(int i = 1; i <= MIN_CUT4D; i++){
		int tmp = pow(2, i);
		
		if(i > 1)
			offset[i]= counter;
		if(i <= max_x){		
			xdiv_size[i] = xdiv_size[0]/tmp; 
			xarr_size[i] = tmp; 
			x *= 2;
		} 
		else{
			xdiv_size[i] = xdiv_size[i-1];
			xarr_size[i] = xarr_size[i-1];  
		}
		if(i <= max_y){		
			ydiv_size[i] = ydiv_size[0]/tmp; 
			yarr_size[i] = tmp; 			
			x *= 2;
		} 
		else{
			ydiv_size[i] = ydiv_size[i-1];
			yarr_size[i] = yarr_size[i-1];  
		}
		if(i <= max_z){		
			zdiv_size[i] = zdiv_size[0]/tmp; 
			zarr_size[i] = tmp; 
			x *= 2;
		} 
		else{
			zdiv_size[i] = zdiv_size[i-1]; 
			zarr_size[i] = zarr_size[i-1]; 
		}
		if(i <= max_t){		
			tdiv_size[i] = tdiv_size[0]/tmp; 
			tarr_size[i] = tmp; 
			x *= 2;
		} 
		else{
			tdiv_size[i] = tdiv_size[i-1]; 
			tarr_size[i] = tarr_size[i-1]; 
		}
		counter += x;
		 
	}
	
#endif	
	
	cout<<"total number of LB elements "<<counter<<endl;
	
	
#if D3D

	for(int i = 1; i <= MIN_CUT; i++){
		int x = pow(2, i);
		counter += x*x*x;
	}
	cout<<counter<<endl;
	lb_array = new int[counter];
	memset((void *)lb_array, 0, counter * sizeof(int));
	
	
#else
	
	
	lb_array4D = new int[counter];
	memset((void *)lb_array4D, 0, counter * sizeof(int));
	
	
			
#endif

	M_BASE[t_index] = M_T_BASE;
	M_BASE[x_index] = M_X_BASE;
	M_BASE[y_index] = M_Y_BASE;
	M_BASE[z_index] = M_Z_BASE;	
	
	S_BASE[t_index] = S_T_BASE;
	S_BASE[x_index] = S_X_BASE;
	S_BASE[y_index] = S_Y_BASE;
	S_BASE[z_index] = S_Z_BASE;	
	
	
	
	CO_THREAD_DATA coThreadData[NUM_THREADS];
	
	
	for (int i = 0; i < NUM_THREADS; i++){
		CO_THREAD_DATA *cd = &coThreadData[i];
		cd->data = dpdata4;
		cd->tstart = 0;
		cd->tend = NUM_THREADS - 1;
		cd->id = i;
		cd->cut = cut;
		cd->depth = 0;
		cd->barrierTree = barrierTree;
		cd->threadWaitTime = 0.0;
		cd->counter = 0;
		
	}
	
	FlushCacheMemory(dpdata4, DATA_SIZE * sizeof(double));
	
	timer->start();
	
	pthread_t threadId[NUM_THREADS];
	for (int i = 0; i < NUM_THREADS; i++){
		pthread_create(&threadId[i], 0, PcoThread, &coThreadData[i]);	
	}
	
	
	
	for (int i = 0; i < NUM_THREADS; i++){
		pthread_join(threadId[i], 0);
		printf("thread %i total sync time: %.4f\n", i, coThreadData[i].threadWaitTime);
	}
	
	float max_wait = coThreadData[0].threadWaitTime;
	float total_wait = coThreadData[0].threadWaitTime;
	
	for(int i = 1; i < NUM_THREADS; i++){
		
		total_wait += coThreadData[i].threadWaitTime;
		
		if(coThreadData[i].threadWaitTime > max_wait)
			max_wait = coThreadData[i].threadWaitTime;
		
	}
	cout<<"Maximum Waiting Time : "<<max_wait<<endl<<"Total Wait : "<<total_wait<<endl;
	printf("Parallel GCO One DP average time: %f\n", timer->elapsed());
	printf("Naive DP average time: %f\n", GetStatTime(tsample, STAT_SAMPLES));
	//printf("ATS1() Error: %e\n", GetAbsoluteDiffDP(dpdata1 + DATA_SIZE * (TIME_STEPS & 1), dpdata2 + DATA_SIZE * (TIME_STEPS & 1), DATA_SIZE));
	//printf("ATS2() Error: %e\n", GetAbsoluteDiffDP(dpdata1 + DATA_SIZE * (TIME_STEPS & 1), dpdata3 + DATA_SIZE * (TIME_STEPS & 1), DATA_SIZE));
	
	printf("3DCO() Error: %e\n", GetAbsoluteDiffDP(dpdata1 + DATA_SIZE * (TIME_STEPS & 1), dpdata4 + DATA_SIZE * (TIME_STEPS & 1), DATA_SIZE));
	
	cout<<"Spinning Time = "<<accum<<endl;
	_mm_free(dpdata1);
	delete[] lb_array4D;
	//_mm_free(dpdata2);
	//_mm_free(dpdata3);
	_mm_free(dpdata4);

	delete timer;

	return 0;
}

#endif
