#include "mat.h"
#include "prng.h"


matrix* create_matrix(mem_arena* arena, u32 rows, u32 cols) {
    matrix* out = PUSH_STRUCT(arena, matrix);
    if (out == NULL) {
        return NULL;
    }

    out->rows = rows;
    out->cols = cols;
    out->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);
    if (out->data == NULL) {
        return NULL;
    }

    return out;
}

matrix* load_data(mem_arena* arena, u32 rows, u32 cols, const char* filename) {
    matrix* mat = create_matrix(arena, rows, cols);
    if (mat == NULL) {
        return NULL;
    }

    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size = MIN(size, sizeof(f32) * rows * cols);
    fread(mat->data, 1, size, f);

    fclose(f);
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

void scale(matrix* mat, f32 x) {
    u64 size = (u64)mat->rows * mat->cols;
    for(u64 i=0; i < size; i++)  {
        mat->data[i] *= x;
    }
}


b32 add(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) { return false; }
    if (out->rows != a->rows || out->cols != a->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

b32 sub(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) { return false; }
    if (out->rows != a->rows || out->cols != a->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

b32 mul(
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

    for (u32 i = 0; i < out->rows; i++) {
        for (u32 k = 0; k < a_cols; k++) {
            for (u32 j = 0; j < out->cols; j++) {
                f32 av = transpose_a ?
                    a->data[i + k * a->cols] :
                    a->data[k + i * a->cols];
                f32 bv = transpose_b ?
                    b->data[k + j * b->cols] :
                    b->data[j + k * b->cols];
                out->data[j + i * out->cols] += av * bv;
            }
        }
    }

    return true;
}

b32 relu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0.0f, in->data[i]);
    }

    return true;
}

b32 softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    f32 max_value = in->data[0];
    for (u64 i = 1; i < size; i++) {
        if (in->data[i] > max_value) {
            max_value = in->data[i];
        }
    }

    f32 total = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i] - max_value);
        total += out->data[i];
    }

    scale(out, 1.0f / total);
    return true;
}

b32 cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }
    if (out->rows != p->rows || out->cols != p->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        f32 q_safe = MAX(q->data[i], 1e-7f);
        out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q_safe);
    }

    return true;
}

b32 relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
    if (out->rows != in->rows || out->cols != in->cols) { return false; }
    if (out->rows != grad->rows || out->cols != grad->cols) { return false; }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }

    return true;
}

b32 softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
) {
    u64 size = (u64)softmax_out->rows * softmax_out->cols;

    for (u64 i = 0; i < size; i++) {
        f32 acc = 0.0f;
        for (u64 j = 0; j < size; j++) {
            f32 local_grad = softmax_out->data[i] *
                ((i == j ? 1.0f : 0.0f) - softmax_out->data[j]);
            acc += local_grad * grad->data[j];
        }
        out->data[i] += acc;
    }
    return true;
}



b32 cross_entropy_add_grad(
        matrix* p_grad, matrix* q_grad,
        const matrix* p, const matrix* q, const matrix* grad
    )  {
    
    u64 size = (u64)p->rows * p->cols;
    
    if(p_grad != NULL) {
        for (u32 i=0; i< size; i++) {
            p_grad->data[i] += -logf(q->data[i] * grad->data[i]);
        }
    }
    
    if(q_grad != NULL) {
        for(u32 i=0; i<size; i++) {
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
        }    
    }

    return true;

}
