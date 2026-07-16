#include "mat.h"
#include "prng.h"

matrix* create_matrix(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);

    return mat;
}

void clear(matrix* mat) {
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void fill(matrix* mat, f32 x) {
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = x;
    }
}

void fill_rand(matrix* mat, f32 lower, f32 upper) {
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }

}

void scale(matrix* mat, f32 scale) {
    u64 size = (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;
    }
}


b32 add(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

b32 sub(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] *
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->rows; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->rows; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

b32 matmul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows) { return false; }
    if (out->rows != a_rows || out->cols != b_cols) { return false; }

    if (zero_out) {
        clear(out);
    }

    u32 transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0: { _mat_mul_nn(out, a, b); } break;
        case 1: { _mat_mul_nt(out, a, b); } break;
        case 2: { _mat_mul_tn(out, a, b); } break;
        case 3: { _mat_mul_tt(out, a, b); } break;
    }

    return true;
}

b32 relu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
    }

    return true;
}

b32 softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;

    f32 max_value = in->data[0];
    for (u64 i = 1; i < size; i++) {
        max_value = MAX(max_value, in->data[i]);
    }

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i] - max_value);
        sum += out->data[i];
    }

    scale(out, 1.0f / sum);

    return true;
}


b32 reinforce_loss(
    matrix* out, const matrix* probs, const matrix* advantages
) {

    if (
        probs->rows != advantages->rows ||
        probs->cols != advantages->cols
    ) { return false; }
    if (
        out->rows != advantages->rows ||
        out->cols != advantages->cols
    ) { return false; }

    u64 size = (u64)out->rows * out->cols;

    for (u64 i = 0; i < size; i++) {
        f32 p = MAX(probs->data[i], 1e-8f);
        out->data[i] = -logf(p) * advantages->data[i];
    } 
    return true;
}


b32 reinforce_add_grad(
    matrix* prob_grads,
    const matrix* probs,
    const matrix* rt,
    const matrix* grad
) {
    if (probs->rows != rt->rows || probs->cols != rt->cols) { return false; }
    if (grad->rows != probs->rows || grad->cols != probs->cols) { return false; }
    if (prob_grads->rows != probs->rows || prob_grads->cols != probs->cols) { return false; }

    u64 size = (u64)probs->rows * probs->cols;
    for (u64 i = 0; i < size; i++) {
        f32 p = MAX(probs->data[i], 1e-8f);
        prob_grads->data[i] += grad->data[i] * (-rt->data[i] / p);
    }
    return true;

}

b32 relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    if (out->rows != grad->rows || out->cols != grad->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }

    return true;
}


// b32 softmax_add_grad(
//     matrix* out, const matrix* softmax_out, const matrix* grad
// ) {
//     if (out->rows != softmax_out->rows || out->cols != softmax_out->cols) {
//         return false;
//     }
//     if (grad->rows != softmax_out->rows || grad->cols != softmax_out->cols) {
//         return false;
//     }

//     u64 size = (u64)softmax_out->rows * softmax_out->cols;
//     f32 dot = 0.0f;
//     for (u64 i = 0; i < size; i++) {
//         dot += grad->data[i] * softmax_out->data[i];
//     }

//     for (u64 i = 0; i < size; i++) {
//         out->data[i] += softmax_out->data[i] * (grad->data[i] - dot);
//     }

//     return true;
// }

b32 softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
) {
    u64 size = (u64)softmax_out->cols * softmax_out->rows;
    f32 dot = 0.0f;
    for(u64 i=0; i<size; i++) {
        dot += grad->data[i] * softmax_out->data[i];
    }

    for(u64 i=0; i<size; i++) {
        out->data[i] += softmax_out->data[i] * (grad->data[i] - dot);
    }

    return true;
    
}
