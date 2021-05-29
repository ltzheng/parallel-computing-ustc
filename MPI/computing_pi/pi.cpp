#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
    static long num_steps = 10000;
    double step;
    double x, pi, sum;
    double startTime, endTime;
    int numProcess, processId;

    MPI_Init(&argc, &argv); // 并行环境初始化, 通过argc, argv得到命令行参数
    MPI_Comm_rank(MPI_COMM_WORLD, &processId); // 得到本进程在通信空间中的rank值(0到p-1), 相当于进程id
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess); // 获得进程个数

    if (processId == 0) // root进程
    {
        cout << "Number of parallel processes:" << numProcess << endl;
        startTime = MPI_Wtime();
    }

    MPI_Bcast(&num_steps, 1, MPI_INT, 0, MPI_COMM_WORLD); // 多组通信
    // 从标识为0的进程将数量为1的消息广播发送到组内的所有其它的进程(包括本身)
    step = 1.0 / (double)num_steps;
    sum = 0.0;
    for (int i = processId; i < num_steps; i = i + numProcess)
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    sum = sum * step;

    // 将各进程的同一个变量参与规约计算，并向指定的进程输出计算结果
    // 输入地址&sum, 输出地址&pi, 数据尺寸1, 数据类型DOUBLE, 规约类型SUM, 目标进程为0
    MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (processId == 0) // root进程
    {
        endTime = MPI_Wtime();
        cout << "Elapsed Time: " << (endTime - startTime) << endl;
        cout << "pi: " << pi << endl;
    }
    // MPI_Init到MPI_Finalize之间的代码在每个进程中都会被执行一次
    MPI_Finalize(); // 并行代码的结束, 结束除主进程外其它进程
    return 0;
}
