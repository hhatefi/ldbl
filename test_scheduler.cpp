/* 
 * File:   test_schedule.cpp
 * Author: hatefi
 *
 * Created on October 9, 2010, 12:53 PM
 */

#include <cstdlib>
#include <math.h>
#include <iostream>
#include "scheduler.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    int N, M;
    cin>>N>>M;
    int *points = new int[N];
    //
    for(int i =0; i < N; i++)
        cin>>points[i];

    AllocationTable *table =  balanceLoad(points, N, M);

    AllocationTable *traverser = table;

    while(traverser)
    {
        for(int i = 0; i < N; i++)
            cout<<i<<" ("<<traverser -> allocationList[i]<<"), ";
        cout<<endl;

        traverser = traverser -> nextList;
    }

    delete table;
    delete []points;
    
    return 0;
}

