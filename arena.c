mem_arena* arena_create(u64 capacity) {
    mem_arena* arena = malloc(sizeof(mem_arena));
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

void arena_destroy(mem_arena* arena) {
    if (arena == NULL) {
        return;
    }
    free(arena->memory);
    free(arena);
}

void* arena_push(mem_arena* arena, u64 size, b32 zero) {
    if (arena == NULL) {
        return NULL;
    }

    u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
    if (pos_aligned > arena->capacity || size > arena->capacity - pos_aligned) {
        return NULL;
    }

    u64 new_pos = pos_aligned + size;
    u8* out = arena->memory + pos_aligned;
    arena->pos = new_pos;

    if (zero) {
        memset(out, 0, size);
    }

    return out;
}
