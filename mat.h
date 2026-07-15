#ifndef MAT_H
#define MAT_H

#include "base.h"
#include "arena.h"

typedef struct {
    u32 rows, cols;
    // Row-major
    f32* data;
} matrix;

matrix* create_matrix(mem_arena* arena, u32 rows, u32 cols);
void clear(matrix* mat);
void fill(matrix* mat, f32 x);
void fill_rand(matrix* mat, f32 lower, f32 upper);
void scale(matrix* mat, f32 scale);
b32 add(matrix* out, const matrix* a, const matrix* b);
b32 sub(matrix* out, const matrix* a, const matrix* b);
b32 mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);
b32 relu(matrix* out, const matrix* in);
b32 softmax(matrix* out, const matrix* in);
b32 reinforce_loss(matrix* out, const matrix* probs, const matrix* advantages);
b32 reinforce_add_grad(
    matrix* prob_grads,
    const matrix* probs,
    const matrix* rt,
    const matrix* grad
);
b32 relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
);

#endif
