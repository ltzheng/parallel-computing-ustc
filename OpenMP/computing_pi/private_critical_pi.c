#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2
static long num_steps = 100000;
double step;

//thread 0 iterates step 0,2,4,...
//thread 1 iterates step 1,3,5,...
/*
    when critical secition is executed by thread 0,
    if thread 1 arrived here, it will be blocked until
    thread 0 exits the critival section
*/
void main()
{
    int i;
    double pi = 0.0;
    double sum = 0.0;
    double x = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(i, x, sum) 
//variable x and sum are private for each thread
{
    int id;
    id = omp_get_thread_num();
    for (i = id, sum = 0.0; i < num_steps; i = i + NUM_THREADS)
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
#pragma omp critical 
//code here can only be executed by one thread at a time
    pi += sum * step;
}
    printf("%lf\n",pi);
}
