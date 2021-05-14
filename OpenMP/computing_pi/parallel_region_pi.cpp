#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
static long num_steps = 100000;
double step;

//thread 0 iterates step 0,2,4,...
//thread 1 iterates step 1,3,5,...
void main()
{
    int i;
    double x, pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS); //set 2 threads
#pragma omp parallel private(i) // parallel region begins
{
    double x;
    int id;
    id = omp_get_thread_num();
    for (i = id, sum[id] = 0.0; i < num_steps; i = i + NUM_THREADS)
    {
        x = (i + 0.5) * step;
        sum[id] += 4.0 / (1.0 + x * x);
    }
}
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
    {
        pi += sum[i] * step;
    }
    printf("%lf\n", pi);
}
