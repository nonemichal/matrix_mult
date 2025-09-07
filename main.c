#include "cnpy.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_A "matrix_A.npy"
#define MATRIX_B "matrix_B.npy"

#define TRANSPOSE_MODE true

typedef struct {
  size_t rows;
  size_t cols;
  double *data;
} Matrix;

static inline double get_val_2d(const Matrix *matrix, size_t row, size_t col) {
  return matrix->data[row * matrix->cols + col];
}

static inline double get_val_1d(const Matrix *matrix, size_t idx) {
  return matrix->data[idx];
}

static inline void set_val_2d(Matrix *matrix, double val, size_t row,
                              size_t col) {
  matrix->data[row * matrix->cols + col] = val;
}

static inline void set_val_1d(Matrix *matrix, double val, size_t idx) {
  matrix->data[idx] = val;
}

static inline Matrix new_matrix(size_t rows, size_t cols) {
  double *data = (double *)malloc(rows * cols * sizeof(double));
  return (Matrix){rows, cols, data};
}

Matrix from_npy_array(const cnpy_array *npy_array) {
  size_t rows = npy_array->dims[0];
  size_t cols = npy_array->dims[1];
  Matrix matrix = new_matrix(rows, cols);

  size_t npy_index[] = {0, 0};

  for (npy_index[0] = 0; npy_index[0] < rows; npy_index[0]++) {
    for (npy_index[1] = 0; npy_index[1] < cols; npy_index[1]++) {
      double val = cnpy_get_f8(*npy_array, npy_index);
      set_val_2d(&matrix, val, npy_index[0], npy_index[1]);
    }
  }

  return matrix;
}

Matrix transpose(const Matrix *original) {
  /* There is no possibility to transpose rectangular matrix with swap
   * (MxN != NxM) so new one is allocated */
  size_t rows = original->cols;
  size_t cols = original->rows;

  Matrix transposed = new_matrix(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      double val = get_val_2d(original, j, i);
      set_val_2d(&transposed, val, i, j);
    }
  }

  return transposed;
}

double dot(const Matrix *a, const Matrix *b, size_t idx_row_a,
           size_t idx_col_b) {
  size_t k = a->cols;   // common dimension of matrices
  double product = 0.0; // result dot product

  for (size_t i = 0; i < k; i++) {
    double a_val = get_val_2d(a, idx_row_a, i);
    double b_val = 0;

#if TRANSPOSE_MODE
    b_val = get_val_2d(b, idx_col_b, i);
#else
    b_val = get_val_2d(b, i, idx_col_b);
#endif

    product += a_val * b_val;
  }

  return product;
}

Matrix mult(const Matrix *a, const Matrix *b) {
  assert(a->cols == b->rows);

  size_t rows = a->rows;
  size_t cols = b->cols;

  Matrix res_matrix = new_matrix(rows, cols);
  Matrix b_used = {0};

#if TRANSPOSE_MODE
  b_used = transpose(b);
  printf("Transposed B matrix: rows -> %ld, cols -> %ld\n", b_used.rows,
         b_used.cols);
#else
  b_used = *b;
#endif

  for (size_t current_row = 0; current_row < rows; current_row++) {
    for (size_t current_col = 0; current_col < cols; current_col++) {
      double dot_val = dot(a, &b_used, current_row, current_col);
      set_val_2d(&res_matrix, dot_val, current_row, current_col);
    }
  }

#if TRANSPOSE_MODE
  free(b_used.data);
#endif

  return res_matrix;
}

void print_content(const Matrix *matrix, size_t first_elements) {
  for (size_t i = 0; i < first_elements; i++) {
    double val = matrix->data[i];
    printf("%.4lf\n", val);
  }
}

double time_diff(struct timespec start, struct timespec end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
  struct timespec start, end;

  char *fn1 = MATRIX_A;
  char *fn2 = MATRIX_B;

  cnpy_array a;
  cnpy_array b;

  bool writable = false;

  cnpy_open(fn1, writable, &a);
  cnpy_open(fn2, writable, &b);

  Matrix a_matrix = from_npy_array(&a);
  printf("Created A matrix: rows -> %ld, cols -> %ld\n", a_matrix.rows,
         a_matrix.cols);
  print_content(&a_matrix, 10);

  Matrix b_matrix = from_npy_array(&b);
  printf("Created B matrix: rows -> %ld, cols -> %ld\n", b_matrix.rows,
         b_matrix.cols);
  print_content(&b_matrix, 10);

  clock_gettime(CLOCK_MONOTONIC, &start);
  Matrix c_matrix = mult(&a_matrix, &b_matrix);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Created C matrix: rows -> %ld, cols -> %ld\n", c_matrix.rows,
         c_matrix.cols);
  print_content(&c_matrix, 10);

  printf("Execution time: %.6f s\n", time_diff(start, end));

  free(a_matrix.data);
  free(b_matrix.data);
  free(c_matrix.data);

  return 0;
}
