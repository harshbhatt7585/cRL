#ifndef ARENA_H
#define ARENA_H

#include "base.h"

#define ARENA_ALIGN (sizeof(void*))

typedef struct {
    u8* memory;
    u64 capacity;
    u64 pos;
} mem_arena;

mem_arena* arena_create(u64 capacity);
void arena_destroy(mem_arena* arena);
void* arena_push(mem_arena* arena, u64 size, b32 zero);

#define PUSH_STRUCT(arena, T) \
    (T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) \
    (T*)arena_push((arena), sizeof(T) * (n), true)
#define PUSH_ARRAY_UNINIT(arena, T, n) \
    (T*)arena_push((arena), sizeof(T) * (n), false)

#endif
