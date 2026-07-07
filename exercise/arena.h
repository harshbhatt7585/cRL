#ifndef ARENA_H
#define ARENA_H

#include "base.h"

#define ARENA_ALIGN (sizeof(void*))

typedef struct {
    u8* memory;
    u64 capacity;
    u64 pos;
} mem_arena;

typedef struct {
    mem_arena* arena;
    u64 start_pos;
} mem_arena_temp;

static inline mem_arena* arena_create(u64 capacity) {
    mem_arena* arena = malloc(sizeof(*arena));
    if (arena == NULL) {
        return NULL;
    }

    arena->memory = malloc(capacity);
    if (arena->memory == NULL) {
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->pos = 0;
    return arena;
}

static inline void arena_destroy(mem_arena* arena) {
    if (arena == NULL) {
        return;
    }

    free(arena->memory);
    free(arena);
}

static inline void* arena_push(mem_arena* arena, u64 size, b32 non_zero) {
    u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
    u64 new_pos = pos_aligned + size;

    if (new_pos > arena->capacity) {
        return NULL;
    }

    u8* out = arena->memory + pos_aligned;
    arena->pos = new_pos;

    if (!non_zero) {
        memset(out, 0, size);
    }

    return out;
}

#define PUSH_STRUCT(arena, T) (T*)arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T) (T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), false)
#define PUSH_ARRAY_NZ(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), true)

#endif
