#ifndef MAT_H
#define MAT_H

#include "base.h"
#include "arena.h"

typedef struct {
    u32 rows;
    u32 cols;
    f32* data;
} matrix;

static inline matrix* create_matrix(mem_arena* arena, u32 rows, u32 cols) {
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

#endif
