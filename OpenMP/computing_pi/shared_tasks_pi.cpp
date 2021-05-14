#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
static long num_steps = 100000;
double step;

//thread 0 iterates step [0, num_steps/2-1]
//thread 1 iterates step [num_steps/2, num_steps-1]
void main()
{
    int i;
    double x, pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);  //set 2 threads
#pragma omp parallel // parallel region begins
{
    double x;
    int id;
    id = omp_get_thread_num();
    sum[id] = 0;
#pragma omp for 
//chunk unspecified, iteration assigned to each thread averagely and continuously
    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum[id] += 4.0 / (1.0 + x * x);
    }
}
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
    {
        pi += sum[i] * step;
    }
    printf("%lf\n",pi);
}