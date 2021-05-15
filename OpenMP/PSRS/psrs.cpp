#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#define NUM_THREADS 4

int p = 4;
int n = 160; //length of the array
std::vector<int> merge2DArrays(std::vector<std::vector<int>> arrays, int l, int r);
std::vector<int> merge(std::vector<int> a, std::vector<int> b);


int main()
{
    int i;
    int num = n / p;
    int array[n];
    for (int i = 0; i < n; i++)
    {
        array[i] = rand() % (INT16_MAX - 0) + 0;
    }
    int result[n];
    int sample[p * p];
    int sampleInterval = num / p;
    int pivots[p - 1];
    int divide_index[p][p] = {0};
    omp_set_num_threads(NUM_THREADS);

    int leftPointer[p][p];
    int arrayProcessor[p][p] = {0};

#pragma omp parallel private(i)
{
    int processorId = omp_get_thread_num();
    //averagely divide & locally sort
    std::sort(array + processorId * num, array + (processorId + 1) * num);
#pragma omp for
    //sample
    for (i = 0; i < p; i++) 
    {
        sample[processorId * p + i] = array[processorId * num + sampleInterval * i];
    }
}
    //sort p^2 samples with 1 processor
    std::sort(sample, sample + p * p);
    //select p-1 pivots with 1 processor
    for (int j = 0; j < p - 1; j++)
    {
        pivots[j] = sample[(j + 1) * p];
    }

#pragma omp parallel private(i)
{
    int processorId = omp_get_thread_num();
    int pivotNum = 0;
    int segmentLength = 0;
    //divide to p parts with p-1 pivots
    for (i = 0; i < num; i++) 
    {
        if (array[processorId * num + i] <= pivots[pivotNum])
        {
            segmentLength++;
        }
        else if (pivotNum == p)
            break;
        else
        {
            divide_index[processorId][pivotNum] = segmentLength;
            segmentLength = 1;
            pivotNum++;
        }
    }
    divide_index[processorId][p - 1] = num - i + 2;
}
    //set left pointer for global exchange
    int segmentLength = 0;
    for (int pivotNum = 0; pivotNum < p; pivotNum++)
    {
        for (int j = 0; j < p; j++)
        {
            leftPointer[j][pivotNum] = segmentLength;
            segmentLength += divide_index[j][pivotNum];
        }
    }
#pragma omp parallel private(i)
{
    int processorId = omp_get_thread_num();
    int count = 0;
    for (i = 0; i < p; i++)
    {
        for (int j = 0; j < divide_index[processorId][i]; j++)
        {
            result[leftPointer[processorId][i] + j] = array[processorId * num + count];
            count++;
        }
    }
#pragma omp barrier
    //merge sort
    std::vector<std::vector<int>> exchangedArray;
    for (int k = 0; k < p - 1; k++)
    {
        std::vector<int> temp(result + leftPointer[k][processorId], result + leftPointer[k + 1][processorId]);
        exchangedArray.push_back(temp);
    }
    if (processorId == p - 1)
    {
        std::vector<int> temp(result + leftPointer[p - 1][p - 1], result + n);
        exchangedArray.push_back(temp);
    }
    else
    {
        std::vector<int> temp(result + leftPointer[p - 1][processorId], result + leftPointer[0][processorId + 1]);
        exchangedArray.push_back(temp);
    }
    std::vector<int> res = merge2DArrays(exchangedArray, 0, p - 1);
    int res_size = res.size();
    int start_pos = leftPointer[0][processorId];
    for (int i = 0; i < res_size; i++)
    {
        array[start_pos + i] = res[i];
    }
}
    //display sort result
    for (int l = 0; l < n; l++)
    {
        printf("%d ", array[l]);
    }
}

std::vector<int> merge2DArrays(std::vector<std::vector<int>> arrays, int l, int r)
{
    if (l == r)
        return arrays[l];
    if (l + 1 == r)
        return merge(arrays[l], arrays[r]);

    int mid = l + (r - l) / 2;
    std::vector<int> left = merge2DArrays(arrays, l, mid);
    std::vector<int> right = merge2DArrays(arrays, mid + 1, r);

    return merge(left, right);
}

std::vector<int> merge(std::vector<int> a, std::vector<int> b)
{
    int l = a.size() + b.size();
    std::vector<int> res(l);

    int i = 0, j = 0;
    for (int k = 0; k < l; k++)
    {
        if (i >= a.size())
            res[k] = b[j++];
        else if (j >= b.size())
            res[k] = a[i++];
        else if (a[i] < b[j])
            res[k] = a[i++];
        else
            res[k] = b[j++];
    }
    return res;
}