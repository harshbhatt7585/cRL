#ifndef MAT_H
#define MAT_H

#include "base.h"
#include "arena.h"

typedef struct {
    u32 rows;
    u32 cols;
    f32* data;
} matrix;

matrix* create_matrix(mem_arena* arena, u32 rows, u32 cols);
matrix* load_data(mem_arena* arena, u32 rows, u32 cols, const char* filename);
void clear(matrix* mat);
void fill(matrix* mat, f32 x);
void fill_rand(matrix* mat, f32 lower, f32 upper);
void scale(matrix* mat, f32 x);
b32 add(matrix* out, const matrix* a, const matrix* b);
b32 sub(matrix* out, const matrix* a, const matrix* b);
b32 mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);
b32 relu(matrix* out, const matrix* in);
b32 softmax(matrix* out, const matrix* in);
b32 cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
);
b32 cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
);

#endif
