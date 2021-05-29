/*
    GPU上矩阵乘法
*/

//主机端函数
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace chrono;
//CUDA RunTime API
#include <cuda_runtime.h>
//单个 block 大小
#define THREAD_NUM 256
//矩阵大小
#define ROW_A 100
#define COL_A 100
#define ROW_B 100
#define COL_B 100
#define RANGE 10
//block个数
int blocks_num = (ROW_A * COL_B + THREAD_NUM - 1) / THREAD_NUM + 1;

//生成矩阵
void generateMatrix(float *A, float *B)
{
    srand((int)time(NULL));
    for (int i = 0; i < ROW_A; i++)
    {
        for (int j = 0; j < COL_A; j++)
        {
            *(A + i * COL_A + j) = (rand() / float(RAND_MAX)) * RANGE;
        }
    }
    for (int i = 0; i < ROW_B; i++)
    {
        for (int j = 0; j < COL_B; j++)
        {
            *(B + i * COL_B + j) = (rand() / float(RAND_MAX)) * RANGE;
        }
    }
}

//CPU串行版本矩阵乘法
void matMulCPU(float *A, float *B, float *C)
{
    for (int i = 0; i < ROW_A; i++)
    {
        for (int j = 0; j < COL_B; j++)
        {
            float sum = 0;
            for (int k = 0; k < COL_A; k++)
            {
                sum += A[i * COL_A + k] * B[k * COL_A + j];
            }
            C[i * ROW_A + j] = sum;
        }
    }
}

void displayMat(float* A, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << A[i * row + j] << " ";
        }
        cout << endl;
    }
}

//设备端函数
__global__ static void CUDAkernel(const float *a, const float *b, float *c)
{
    //block内的threadID
    const int tid = threadIdx.x;
    //blockID
    const int bid = blockIdx.x;
    //全局threadID
    const int idx = bid * THREAD_NUM + tid;
    const int row = idx / COL_B;
    const int column = idx % COL_B;
    //计算矩阵乘法
    if (row < ROW_A && column < COL_B)
    {
        float t = 0;
        for (int i = 0; i < COL_A; i++)
        {
            t += a[row * COL_A + i] * b[i * COL_B + column];
        }
        c[row * COL_B + column] = t;
    }
}

int main()
{
    //定义矩阵
    float *a, *b, *c, *d;
    //分配主机端内存
    a = (float *)malloc(sizeof(float) * ROW_A * COL_A);
    b = (float *)malloc(sizeof(float) * ROW_B * COL_B);
    c = (float *)malloc(sizeof(float) * ROW_A * COL_B);
    d = (float *)malloc(sizeof(float) * ROW_A * COL_B);
    float *cuda_a, *cuda_b, *cuda_c;
    //分配设备端显存
    cudaMalloc((void **)&cuda_a, sizeof(float) * ROW_A * COL_A);
    cudaMalloc((void **)&cuda_b, sizeof(float) * ROW_B * COL_B);
    cudaMalloc((void **)&cuda_c, sizeof(float) * ROW_A * COL_B);
    //生成矩阵 a, b
    generateMatrix(a, b);
    cout << "Matrix A:" << endl;
    displayMat(a, ROW_A, COL_A);
    cout << "Matrix B:" << endl;
    displayMat(b, ROW_B, COL_B);
    //开始计算并行时间
    auto par_start = system_clock::now();
    //cudaMemcpyHostToDevice 从内存复制到显存
    //cudaMemcpyDeviceToHost 从显存复制到内存
    cudaMemcpy(cuda_a, a, sizeof(float) * ROW_A * COL_A, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float) * ROW_B * COL_B, cudaMemcpyHostToDevice);
    //设备端函数
    CUDAkernel<<<blocks_num, THREAD_NUM, 0>>>(cuda_a, cuda_b, cuda_c);
    //cudaMemcpy 将结果从显存中复制回内存
    //比较加速比
    cudaMemcpy(c, cuda_c, sizeof(float) * ROW_A * COL_B, cudaMemcpyDeviceToHost);
    auto par_end = system_clock::now();

    auto cpu_start = system_clock::now();
    matMulCPU(a, b, d);
    auto cpu_end = system_clock::now();
    
    auto par_time = duration_cast<nanoseconds>(par_end - par_start);
    auto cpu_time = duration_cast<nanoseconds>(cpu_end - cpu_start);
    cout << "Matrix C:" << endl;
    displayMat(c, ROW_A, COL_B);
    cout << "Matrix D:" << endl;
    displayMat(d, ROW_A, COL_B);
    cout << "Parallel time: " << par_time.count() << endl;
    cout << "Cpu time: " << cpu_time.count() << endl;
    cout << "Speedup: " << cpu_time.count() /par_time.count() << endl;
    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
}
