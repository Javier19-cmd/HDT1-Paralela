#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void openmp_hello(int thread_id, int total_threads);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Uso: %s <Número de hilos>\n", argv[0]);
        return 1;
    }

    int thread_count = strtol(argv[1], NULL, 10);

    #pragma omp parallel num_threads(thread_count)
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        openmp_hello(thread_id, total_threads);
    }

    return 0;
}

void openmp_hello(int thread_id, int total_threads) {
    if (thread_id % 2 == 0) {
        printf("Saludos del hilo %d\n", thread_id);
    } else {
        printf("Feliz cumpleaños número %d !\n", thread_id);
    }
}
