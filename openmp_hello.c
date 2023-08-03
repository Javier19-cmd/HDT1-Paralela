#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void openmp_hello(void);

int main(int argc, char* argv[]) {
	int thread_count = strtol(argv[1], NULL, 10);

#pragma omp parallel num_threads(thread_count)
	openmp_hello();

	return 0;
}

void openmp_hello(void) {

	printf("Hola mundo\n");

} /*openmp_hello*/
