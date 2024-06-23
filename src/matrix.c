#include <assert.h>
#include <stdlib.h>
#include "matrix.h"
#include "ds.h"

void matrix_init(Matrix *matrix, unsigned int rows, unsigned int cols) {
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = malloc(rows * cols * sizeof(float));
    if (matrix->data == NULL) {
        DS_PANIC("Buy more RAM!");
    }
}

void matrix_init_eye(Matrix *matrix, unsigned int size) {
    matrix_init(matrix, size, size);
    for (unsigned int i = 0; i < size; i++) {
        matrix_set(matrix, i, i, 1);
    }
}

void matrix_init_random(Matrix *matrix, unsigned int rows, unsigned int cols) {
    matrix_init(matrix, rows, cols);
    for (unsigned int i = 0; i < rows * cols; i++) {
        matrix->data[i] = (float)rand() / RAND_MAX;
    }
}

float matrix_at(const Matrix *matrix, unsigned int row, unsigned int col) {
    unsigned int index = row * matrix->cols + col;
    return matrix->data[index];
}

void matrix_set(Matrix *matrix, unsigned int row, unsigned int col, float value) {
    unsigned int index = row * matrix->cols + col;
    matrix->data[index] = value;
}

#ifdef MATRIX_CPU
void matrix_matmul(const Matrix *m1, const Matrix *m2, Matrix *matrix) {
    assert(m1->cols == m2->rows);
    assert(matrix->rows == m1->rows);
    assert(matrix->cols == m2->cols);
    assert(matrix->data != NULL);

    for (unsigned int row = 0; row < m1->rows; row++) {
        for (unsigned int col = 0; col < m2->cols; col++) {
            float value = 0;
            for (unsigned int t = 0; t < m1->cols; t++) {
                value += matrix_at(m1, row, t) * matrix_at(m2, t, col);
            }
            matrix_set(matrix, row, col, value);
        }
    }
}
#endif
