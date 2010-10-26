/* 
 * File:   scheduler.h
 * Author: hatefi
 *
 * Created on October 8, 2010, 1:37 PM
 */

#ifndef SCHEDULER_H
#define	SCHEDULER_H

class AllocationTable {
public:
    int * allocationList;
    AllocationTable* nextList;
    ~AllocationTable()
    {
        delete [] this -> allocationList;
        if(this -> nextList)
            delete this -> nextList;
    }
};

/*
 *
 */
int compare(int *points, float *optimalSolution, int left, int right)
{
    float leftTimeDown = -1, leftTimeUp = -1, rightTimeDown = -1, rightTimeUp = -1;

    if( floor(optimalSolution[left]) != 0  )
        leftTimeDown = points[left]/floor(optimalSolution[left]);

    if( floor(optimalSolution[right]) != 0  )
        rightTimeDown = points[right]/floor(optimalSolution[right]);

    if( leftTimeDown != -1 && rightTimeDown == -1)
    {
        return 1;
    }

    if( leftTimeDown == -1 && rightTimeDown != -1)
    {
        return -1;
    }

    if( leftTimeDown != -1 && rightTimeDown != -1)
    {
        if( leftTimeDown > rightTimeDown )
            return 1;
        else if ( leftTimeDown < rightTimeDown )
            return -1;
        else
            return 0;
    }

    if( ceil(optimalSolution[left]) != 0  )
        leftTimeUp = points[left]/ceil(optimalSolution[left]);

    if( ceil(optimalSolution[right]) != 0  )
        rightTimeUp = points[right]/ceil(optimalSolution[right]);

    if( leftTimeUp > rightTimeUp )
        return 1;
    else if ( leftTimeUp < rightTimeUp )
        return -1;
    else
        return 0;
}

int isGreater(int *points, float *optimalSolution, int left, int right)
{
    float leftTimeDown = -1, leftTimeUp = -1, rightTimeDown = -1, rightTimeUp = -1;

    if( floor(optimalSolution[left]) != 0  )
        leftTimeDown = points[left]/floor(optimalSolution[left]);

    if( floor(optimalSolution[right]) != 0  )
        rightTimeDown = points[right]/floor(optimalSolution[right]);

    leftTimeUp = points[left]/ceil(optimalSolution[left]);

    rightTimeUp = points[right]/ceil(optimalSolution[right]);

    if( (leftTimeDown > rightTimeDown) || (leftTimeDown == rightTimeDown && leftTimeUp > rightTimeUp) )
    {
       return true;
    }
    else
    {
        return false;
    }
}


AllocationTable *balanceLoad(int *points, int nTiles, int nThreads)
{
    bool *processedTiles = new bool[nTiles];
    float *optimalSolution = new float[nTiles];

    for(int i = 0; i < nTiles; i++)
    {
        processedTiles[i] = false;
    }

    AllocationTable *firstRow = NULL, *currentRow = NULL;
    while (true)
    {
        int totalPoints = 0;
        for(int i = 0; i < nTiles; i++)
            if( !processedTiles[i] )
                totalPoints += points[i];

        // Check the end condition
        if( totalPoints == 0 )
        {
            break;
        }
        else // Allocation of a AllocationTable entry for the current iteration
        {
            // Allocate memory for the current row
            AllocationTable *temp = new AllocationTable;

            // Initialize the first row and/or link the current row to the previous row
            if( firstRow )
            {
                currentRow -> nextList = temp;
                currentRow = temp;
            }
            else /* Only in first iteration*/
            {
                firstRow = currentRow = temp;
            }

        }

        // Finding the optimal solution and sum of its fractional parts
        float sumOfFractionalPart = 0.0;
        for(int i = 0; i < nTiles; i++)
        {
            if( !processedTiles[i] )
            {
                optimalSolution[i] = (float)points[i] / totalPoints * nThreads;
                sumOfFractionalPart += optimalSolution[i] - floor(optimalSolution[i]);
            }
        }

        // Number of rounging up is equal to sum of fractional part in optimal solution
        int numberOfRoundingUp = round(sumOfFractionalPart);


        /*
         * 
         * Here is the sterategy we use to select a solution for rounding up
         *
         */
        currentRow -> allocationList = new int[nTiles];

        for(int i = 0; i < nTiles; i++)
            currentRow -> allocationList[i] = 0;

        for(int  i = 0; i < numberOfRoundingUp; i++)
        {
            int maxIndex = -1;
            for(int j = 0; j < nTiles; j++)
            {
                if( !processedTiles[j] )
                {
                    if( maxIndex == -1 )
                    {
                        maxIndex = j;
                    }
                    else if( isGreater(points, optimalSolution, j, maxIndex ) )
                    {
                        maxIndex = j;
                    }
                }
            }

            if( maxIndex == -1 )
                printf("No Tile find.\n");
            else
            {
                currentRow -> allocationList[maxIndex] = ceil(optimalSolution[maxIndex]);
                processedTiles[maxIndex] = true;
            }

        }

        // Fill out the current row in allocation table

        for(int i = 0; i < nTiles; i++)
        {
            if( !processedTiles[i] )
            {
                currentRow -> allocationList[i] = floor(optimalSolution[i]);
                if( currentRow -> allocationList[i] )
                    processedTiles[i] = true;
            }

        }
    }

    currentRow -> nextList = NULL;
    
    // Delete arrays
    delete []processedTiles;
    delete []optimalSolution;

    return firstRow;
}

#endif	/* SCHEDULER_H */

