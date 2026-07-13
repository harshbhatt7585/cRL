#include "base.h"
#include "arena.h"
#include "prng.h"
#include "mat.h"
#include "autograd.h"

#include "arena.c"
#include "prng.c"
#include "mat.c"
#include "autograd.c"

void create_actor_critic_model(
    mem_arena* arena, model_state* model
) {
    Var* input = var_create(arena, model, 72, 1, VAR_FLAG_NONE);
    model->input = input;

    Var* W0 = var_create(arena, model, 128, 72, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b0 = var_create(arena, model, 128, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    Var* W1 = var_create(arena, model, 128, 128, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b1 = var_create(arena, model, 128, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    Var* W2 = var_create(arena, model, 5, 128, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b2 = var_create(arena, model, 5, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    f32 bound0 = sqrtf(6.0f / (72 + 128));
    f32 bound1 = sqrtf(6.0f / (128 + 128));
    f32 bound2 = sqrtf(6.0f / (128 + 5));
  
    fill_rand(W0->val, -bound0, bound0);
    fill_rand(W1->val, -bound1, bound1);
    fill_rand(W2->val, -bound2, bound2);

    Var* z0_a = var_matmul(arena, model, W0, input, 0);
    Var* z0_b = var_add(arena, model, z0_a, b0, 0);
    Var* z1_a = var_matmul(arena, model, W1, z0_b, 0);
    Var* z1_b = var_add(arena, model, z1_a, b1, 0);
    Var* z2_a = var_matmul(arena, model, W2, z1_b, 0);
    Var* z2_b = var_add(arena, model, z2_a, b2, 0);

    Var* output = var_softmax(arena, model, z2_b, VAR_FLAG_NONE);
    model->output = output;

    Var* returns = var_create(arena, model, 5, 1, VAR_FLAG_NONE);
    model->desired_output = returns;

    Var* cost = var_reinforce_loss(arena, model, output, returns, VAR_FLAG_NONE);
    model->cost = cost;
}


// int main() {
//     mem_arena* arena = arena_create(GiB(1));

//     model_state* model = PUSH_STRUCT(arena, model_state);

//     create_actor_critic_model(arena, model);

//     arena_destroy(arena);

//     return 0;
// }
