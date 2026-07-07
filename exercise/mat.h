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

#endif
