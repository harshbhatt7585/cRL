#include "mat.h"

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



int main(void) {
    mem_arena* arena = arena_create(10000);
    if (arena == NULL) {
        return 1;
    }

    matrix* a = create_matrix(arena, 25, 25);
    if (a == NULL) {
        arena_destroy(arena);
        return 1;
    }

    printf("%f\n", a->data[0]);
    arena_destroy(arena);

    return 0;
}
