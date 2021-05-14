#include <stdio.h>
#include <omp.h>
static long num_steps = 100000;
double step;
#define NUM_THREADS 2

void main()
{
    int i;
    double pi = 0.0;
    double sum = 0.0;
    double x = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for reduction(+: sum) private(x) 
//each thread keeps a private copy of sum
//x is private for each thread
//At last, update global sum by reduction
    for (i = 1; i <= num_steps; i++)
    {
        x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = sum * step;
    printf("%lf\n",pi);
}