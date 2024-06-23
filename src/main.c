#include <stdio.h>
#include "matrix.h"

int main() {
    Matrix m1 = {0};
    Matrix m2 = {0};
    Matrix matrix = {0};

    int n = 5;
    int m = 5;
    int p = 5;

    matrix_init_random(&m1, n, p);
    matrix_init_random(&m2, p, m);
    matrix_init(&matrix, n, m);

    matrix_matmul(&m1, &m2, &matrix);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", matrix_at(&m1, i, j));
        }
        printf("\n");
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f ", matrix_at(&m2, i, j));
        }
        printf("\n");
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f ", matrix_at(&matrix, i, j));
        }
        printf("\n");
    }
}
