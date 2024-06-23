#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    float *data;
    unsigned int rows;
    unsigned int cols;
} Matrix;

void matrix_init(Matrix *matrix, unsigned int rows, unsigned int cols);
void matrix_init_eye(Matrix *matrix, unsigned int size);
void matrix_init_random(Matrix *matrix, unsigned int rows, unsigned int cols);
float matrix_at(const Matrix *matrix, unsigned int row, unsigned int col);
void matrix_set(Matrix *matrix, unsigned int row, unsigned int col, float value);
void matrix_matmul(const Matrix *m1, const Matrix *m2, Matrix *matrix);

#endif // MATRIX_H
