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

int main(int argc, char* argv[]) {
    double integral;
    double a = A, b = B;
    int n = N;
    double h;
    int thread_count = 1; // Número de hilos

    if (argc > 1) {
        a = strtol(argv[1], NULL, 10);
        b = strtol(argv[2], NULL, 10);
        thread_count = strtol(argv[3], NULL, 10); // Leer el número de hilos
    }

    //---- Aproximacion de la integral
    //h = (b-a)/n;
    integral = trapezoides(a, b, n, thread_count); // Pasar thread_count a la función
    printf("Con n = %d trapezoides y %d hilos, nuestra aproximacion\n", n, thread_count);
    printf("de la integral de %f a %f es = %.10f\n", a, b, integral);
    return 0;
}/*main*/

//------------------------------------------
// trapezoides
//
// Estimar la integral mediante sumas de Riemann
// Input: a,b,n,h
// Output: integral
//------------------------------------------

double trapezoides(double a, double b, int n, int thread_count) {
    double integral, h;
    int k;

    //---- Ancho de cada trapezoide
    h = (b - a) / n;

    //---- Valor inicial de la integral (valores extremos)
    integral = (f(a) + f(b)) / 2.0;

    // Paralelizar el bucle utilizando OpenMP
    #pragma omp parallel for num_threads(thread_count) reduction(+: integral)
    for (k = 1; k <= n - 1; k++) {
        integral += f(a + k * h);
    }

    integral = integral * h;
    return integral;
}/*trapezoides*/

//------------------------------------------
// f
//
// Funcion a ser integrada
// Input: x
//------------------------------------------

double f(double x) {
    return x * x;
}
