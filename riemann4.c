/*----------------------------------------------
 * riemann.c - calculo de area bajo la curva
 *----------------------------------------------
 * Sumas de Riemann para calcular la integral f(x)
 *
 * Date:  2021-09-22
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Incluir la biblioteca OpenMP
#define A 1
#define B 40
#define N 10e6

double f(double x);
double trapezoides(double a, double b, int n, int thread_count);

int main(int argc, char *argv[])
{
    double integral;
    double a = A, b = B;
    int n = N;
    double h;
    int thread_count = 1;

    if (argc > 1)
    {
        a = strtol(argv[1], NULL, 10);
        b = strtol(argv[2], NULL, 10);
        thread_count = strtol(argv[3], NULL, 10);
    }

    //---- Aproximacion de la integral
    // h = (b-a)/n;
    integral = trapezoides(a, b, n, thread_count);
    printf("Con n = %d trapezoides y %d hilos, nuestra aproximacion\n", n, thread_count);
    printf("de la integral de %f a %f es = %.10f\n", a, b, integral);

    return 0;
} /*main*/

//------------------------------------------
// trapezoides
//
// Estimar la integral mediante sumas de Riemann
// Input: a,b,n,h
// Output: integral
//------------------------------------------

double trapezoides(double a, double b, int n, int thread_count)
{
    double integral, h;
    int k;

    //---- Ancho de cada trapezoide
    h = (b - a) / n;

    //---- Valor inicial de la integral (valores extremos)
    integral = (f(a) + f(b)) / 2.0;

    // Variables locales
    double n_local, a_local, b_local, suma_local;

    // Suma global como puntero
    double *suma_global;
    suma_global = (double *)malloc(sizeof(double));
    *suma_global = 0.0;

// Paralelizar el bucle utilizando OpenMP
#pragma omp parallel num_threads(thread_count) private(k, n_local, a_local, b_local, suma_local)
    {
        int thread_num = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        printf("ID del Thread: %d de un total de %d hilos\n", thread_num, total_threads);

        // Calcular el rango de trabajo del hilo actual
        int chunk_size = n / total_threads;
        int start = thread_num * chunk_size + 1;
        int end = (thread_num == total_threads - 1) ? n : start + chunk_size - 1;

        n_local = end - start + 1;
        a_local = a + start * h;
        b_local = a_local + (n_local - 1) * h;

        suma_local = 0.0;

        for (k = start; k <= end; k++)
        {
            suma_local += f(a_local + (k - start) * h);
        }

// Proteger la acumulación de suma_local a suma_global
#pragma omp critical
        *suma_global += suma_local;
    }
    // Imprimir suma global
    printf("Suma global: %.10f\n", *suma_global);
    integral = (*suma_global) * h;
    free(suma_global); // Liberar la memoria asignada

    return integral;
} /*trapezoides*/

//------------------------------------------
// f
//
// Funcion a ser integrada
// Input: x
//------------------------------------------

double f(double x)
{
    return x * x;
}
