#include <stdlib.h>
#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std;

int i, j, k;
void PSRS(int *vector, int dim);

int main(int argc, char *argv[])
{
    int dim;
    int numProcess, processId;

    MPI_Init(&argc, &argv);                     // 并行环境初始化, 通过argc, argv得到命令行参数
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);  // 得到本进程在通信空间中的rank值(0到p-1), 相当于进程id
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess); // 获得进程个数

    if (processId == 0) // root进程, 读取向量大小
    {
        cout << "Please input the dimension of vector to sort:" << endl;
        cin >> dim;
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD); // 多组通信
    // 从标识为0的进程将数量为1的消息广播发送到组内的所有其它的进程(包括本身)

    // 维度需要整除处理器个数, 不整除时补0
    int fillZero = numProcess - (dim % numProcess);
    int totalNum = dim + fillZero;

    // 生成随机数组
    int *vector = new int[totalNum];
    srand(0);
    for (int i = 0; i < totalNum; i++)
    {
        if (i < dim)
            vector[i] = rand() % 200;
        else
            vector[i] = 0;
    }
    // 输出随机数组
    if (processId == 0)
    {
        cout << "Number of parallel processes:" << numProcess << endl;
        cout << "Vector to sort:" << endl;
        for (int i = 0; i < dim; i++)
            cout << vector[i] << " ";
        cout << endl;
    }

    // PSRS并行排序
    PSRS(vector, dim);

    return 0;
}

int cmp(const void *a, const void *b)
{
    if (*(int *)a < *(int *)b)
        return -1;
    if (*(int *)a > *(int *)b)
        return 1;
    else
        return 0;
}

void Merge(int *partitions, int *partitionSizes, int numProcess, int processId, int *vector)
{
    int *sortedSubList;
    int *recvDisp, *indexes, *endIndex, *subListSizes, startIndex;

    // 全局交换后每个partition开始的位置
    indexes = (int*)malloc(numProcess * sizeof(int));
    indexes[0] = 0;

    // 每个处理器的子列表在整个vector中的坐标
    startIndex = partitionSizes[0]; 
    endIndex = (int*)malloc(numProcess * sizeof(int));
    
    for (i = 1; i < numProcess; i++)
    {
        startIndex += partitionSizes[i];
        indexes[i] = indexes[i - 1] + partitionSizes[i - 1];
        endIndex[i - 1] = indexes[i];
    }
    endIndex[numProcess - 1] = startIndex;

    sortedSubList = (int*)malloc(startIndex * sizeof(int));
    subListSizes = (int*)malloc(numProcess * sizeof(int));
    recvDisp = (int*)malloc(numProcess * sizeof(int));

    // 归并排序
    for (i = 0; i < startIndex; i++)
    {
        int lowest = __INT32_MAX__;
        int ind = -1;
        for (j = 0; j < numProcess; j++)
        {
            if ((indexes[j] < endIndex[j]) && (partitions[indexes[j]] < lowest))
            {
                lowest = partitions[indexes[j]];
                ind = j;
            }
        }
        sortedSubList[i] = lowest;
        indexes[ind] += 1;
    }

    // 收集各子进程中列表的大小到root进程中
    MPI_Gather(&startIndex, 1, MPI_INT, subListSizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 计算根进程上的相对于recvbuf的偏移量
    if (processId == 0)
    {
        recvDisp[0] = 0;
        for (i = 1; i < numProcess; i++)
            recvDisp[i] = subListSizes[i - 1] + recvDisp[i - 1];
    }

    // 发送各排好序的子列表到root进程
    MPI_Gatherv(sortedSubList, startIndex, MPI_INT, vector, subListSizes, recvDisp, MPI_INT, 0, MPI_COMM_WORLD);

    free(endIndex);
    free(sortedSubList);
    free(indexes);
    free(subListSizes);
    free(recvDisp);
    return;
}

void PSRS(int *vector, int dim)
{
    double startTime = MPI_Wtime();
    int numProcess, processId, *partitionSizes, *newPartitionSizes;
    int subArraySize, startIndex, endIndex, *sample, *newPartitions;
    int dimLocal = dim / numProcess;
    int step = dimLocal / numProcess;

    MPI_Comm_rank(MPI_COMM_WORLD, &processId); // 获得进程个数
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess); // 得到本进程在通信空间中的rank值(0到p-1), 相当于进程id

    // 存储选取的p^2个样本, 用于串行排序
    sample = (int*)malloc(numProcess * numProcess * sizeof(int));
    // 记录按主元划分的块大小
    partitionSizes = (int*)malloc(numProcess * sizeof(int));
    newPartitionSizes = (int*)malloc(numProcess * sizeof(int));

    for (k = 0; k < numProcess; k++)
        partitionSizes[k] = 0;

    startIndex = processId * dim / numProcess;
    if (numProcess == (processId + 1))
        endIndex = dim;
    else
        endIndex = (processId + 1) * dim / numProcess;
    subArraySize = endIndex - startIndex;

    MPI_Barrier(MPI_COMM_WORLD); // 阻塞所有进程直到都调用了它

    // 调用串行排序算法进行局部排序
    qsort(vector + startIndex, subArraySize, sizeof(vector[0]), cmp);

    // 从有序子序列中选取样本
    for (i = 0; i < numProcess; i++)
        sample[processId * numProcess + i] = *(vector + (processId * dimLocal + i * step));

    // p-1个主元
    int *pivot_number = (int*)malloc((numProcess - 1) * sizeof(sample[0])); //主元
    int index = 0;

    MPI_Barrier(MPI_COMM_WORLD); // 阻塞所有进程直到都调用了它

    if (processId == 0)
    {
        // 使用一台处理器进行样本串行排序
        qsort(sample, numProcess * numProcess, sizeof(sample[0]), cmp); //对正则采样的样本进行排序
        // 用一台处理器选择主元
        for (i = 0; i < (numProcess - 1); i++)
            pivot_number[i] = sample[(((i + 1) * numProcess) + (numProcess / 2)) - 1];
    }
    // 将从排好序的样本序列中选取的p-1个主元, 播送给其他处理器
    MPI_Bcast(pivot_number, numProcess - 1, MPI_INT, 0, MPI_COMM_WORLD); // 多组通信
    // 从标识为0的进程将数量为numProcess-1的消息广播发送到组内的所有其它的进程(包括本身)

    // 主元划分, 按p-1个主元划分为p段
    for (i = 0; i < subArraySize; i++)
    {
        if (vector[startIndex + i] > pivot_number[index])
            index ++;
        if (index == numProcess)
        {
            partitionSizes[numProcess - 1] = subArraySize - i + 1;
            break;
        }
        partitionSizes[index]++;
    }
    free(pivot_number);

    int totalSize = 0;
    // sendDisp/recvDisp数组中的每个元素代表了要发送/接收的那块数据相对于缓冲区起始位置的位移量
    int *sendDisp = (int*)malloc(numProcess * sizeof(int));
    int *recvDisp = (int*)malloc(numProcess * sizeof(int));

    // 全局交换: 各处理器将其有序段按段号交换到对应的处理器中
    // 每个进程都向其它所有进程发送消息, 同时都从其它所有进程接收消息
    MPI_Alltoall(partitionSizes, 1, MPI_INT, newPartitionSizes, 1, MPI_INT, MPI_COMM_WORLD);

    // 计算划分的总大小, 给新划分分配空间
    for (i = 0; i < numProcess; i++)
        totalSize += newPartitionSizes[i];
    newPartitions = (int*)malloc(totalSize * sizeof(int));

    sendDisp[0] = 0;
    recvDisp[0] = 0;
    for (i = 1; i < numProcess; i++)
    {
        sendDisp[i] = partitionSizes[i - 1] + sendDisp[i - 1];
        recvDisp[i] = newPartitionSizes[i - 1] + recvDisp[i - 1];
    }

    // MPI_Alltoallv先告诉每个节点需要接收多少数据, 由此确定接收缓冲区大小
    // partitionSizes/newPartitionSizes中的元素代表往其他节点各发送/接收多少数据
    MPI_Alltoallv(&(vector[startIndex]), partitionSizes, sendDisp, MPI_INT, newPartitions, newPartitionSizes, recvDisp, MPI_INT, MPI_COMM_WORLD);
    free(sendDisp);
    free(recvDisp);

    // 归并排序
    Merge(newPartitions, newPartitionSizes, numProcess, processId, vector);

    double endTime = MPI_Wtime();
    // 输出排序后结果
    if (processId == 0)
    {
        cout << "Result:" << endl;
        for (int i = 0; i < dim; i++)
        {
            cout << vector[i] << " ";
        }
        cout << endl;
        cout << "Elapsed Time: " << (endTime - startTime) << endl;
    }
    if (numProcess > 1)
        free(newPartitions);

    free(partitionSizes);
    free(newPartitionSizes);
    free(sample);
    free(vector);
    // MPI_Init到MPI_Finalize之间的代码在每个进程中都会被执行一次
    MPI_Finalize(); // 并行代码的结束, 结束除主进程外其它进程
}