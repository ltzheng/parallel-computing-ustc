# Parallel Computing Course at USTC

## Lab 1

### Computing $\pi$

1) serial

    [serial_pi.c](/OpenMP/computing_pi/serial_pi.c)

    ```bash
    gcc serial_pi.c -o serial_pi
    ./serial_pi
    ```

2) parallel

    see [parallel_region_pi.c](/OpenMP/computing_pi/parallel_region_pi.c)

    ```bash
    gcc parallel_region_pi.c -fopenmp -o parallel_region_pi
    ./parallel_region_pi
    ```

3) parallel with shared tasks

    see [shared_tasks_pi.c](/OpenMP/computing_pi/shared_tasks_pi.c)

    ```bash
    gcc shared_tasks_pi.c -fopenmp -o shared_tasks_pi
    ./shared_tasks_pi
    ```

5) parallel with private variables & critical section

    see [private_critical_pi.c](/OpenMP/computing_pi/private_critical_pi.c)

    ```bash
    gcc private_critical_pi.c -fopenmp -o private_critical_pi
    ./private_critical_pi
    ```

4) parallel with reduction

    see [parallel_reduction_pi.c](/OpenMP/computing_pi/parallel_reduction_pi.c)

    ```bash
    gcc parallel_reduction_pi.c -fopenmp -o parallel_reduction_pi
    ./parallel_reduction_pi
    ```

### PSRS Sorting