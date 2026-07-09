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

static inline void arena_pop(mem_arena* arena, u64 size) {
    size = MIN(size, arena->pos);
    arena->pos -= size;
}

static inline void arena_pop_to(mem_arena* arena, u64 pos) {
    u64 size = pos < arena->pos ? arena->pos - pos : 0;
    arena_pop(arena, size);
}

static inline mem_arena_temp arena_temp_begin(mem_arena* arena) {
    return (mem_arena_temp) {
        .arena = arena,
        .start_pos = arena->pos,
    };
}

static inline void arena_temp_end(mem_arena_temp temp) {
    arena_pop_to(temp.arena, temp.start_pos);
}

static mem_arena* _scratch_arenas[2] = { NULL, NULL };

static inline mem_arena_temp arena_scratch_get(mem_arena** conflicts, u32 num_conflicts) {
    i32 scratch_index = -1;

    for (i32 i = 0; i < 2; i++) {
        b32 conflict_found = false;

        for (u32 j = 0; j < num_conflicts; j++) {
            if (_scratch_arenas[i] == conflicts[j]) {
                conflict_found = true;
                break;
            }
        }

        if (!conflict_found) {
            scratch_index = i;
            break;
        }
    }

    if (scratch_index == -1) {
        return (mem_arena_temp){ 0 };
    }

    if (_scratch_arenas[scratch_index] == NULL) {
        _scratch_arenas[scratch_index] = arena_create(MiB(1));
    }

    return arena_temp_begin(_scratch_arenas[scratch_index]);
}

static inline void arena_scratch_release(mem_arena_temp scratch) {
    arena_temp_end(scratch);
}

#define PUSH_STRUCT(arena, T) (T*)arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T) (T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), false)
#define PUSH_ARRAY_NZ(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), true)

#endif
