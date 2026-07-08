#include "mat.h"

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
