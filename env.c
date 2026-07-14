// Env will define game dynamics and add reward structure
#include <stdlib.h>
#include "base.h"

#include "base.h"
#include "arena.h"
#include "prng.h"
#include "mat.h"
#include "autograd.h"
#include "model.c"

u32 randn(u32 range) {
    return arc4random_uniform(range);
}

typedef enum  {
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3,
    NONE = 4
} ACTION;

typedef struct {
    i32 x;
    i32 y;
} State;

typedef struct {
    State snake;
    State food;
    f32 score;
    u32 foods_eaten;
    
    u32 rows;
    u32 cols;
    u32 grid_size;
    u64 steps;
    ACTION pov;
} SnakeENV;


#define BUFFER_SIZE 1024
#define EPISODE_LEN 100

typedef struct {
    State states[EPISODE_LEN];
    State food_states[EPISODE_LEN];
    ACTION povs[EPISODE_LEN];
    ACTION actions[EPISODE_LEN];
    f32 rewards[EPISODE_LEN];
    f32 returns[EPISODE_LEN];
    State next_states[EPISODE_LEN];
    b32 dones[EPISODE_LEN];

    u32 len;

} Trajactory;

typedef struct {
    Trajactory trajactories[BUFFER_SIZE];
    u32 count;
} ReplayBuffer;


SnakeENV* create_env(
    u32 grid_size
) {
    SnakeENV* env = malloc(sizeof(*env));
    if(env==NULL) return NULL;

    u32 side = (u32)sqrtf((f32)grid_size);

    if (side * side != grid_size) {
        return NULL;
    }

    env->snake = (State){ .x = 0, .y = 0 };
    env->food = (State){ .x = 5, .y = 5 };
    env->rows = side;
    env->cols = side;
    env->grid_size = grid_size;
    env->score = 0.0f;
    env->foods_eaten = 0;
    env->pov = RIGHT;
    env->steps = 0;
    return env;
}

State get_random_food_loc(u32 cols, u32 rows) {
    return (State) {
        .x = (i32)randn(cols),
        .y = (i32)randn(rows),
    };
}


b32 game_over(SnakeENV* env) {
    return env->snake.x < 0 || env->snake.x >= (i32)env->cols ||
           env->snake.y < 0 || env->snake.y >= (i32)env->rows;
}


f32 get_reward(SnakeENV* env) {
    f32 reward = 0.0f;

    if (env->snake.x == env->food.x && env->snake.y == env->food.y) {
        reward += 20.0f;
        env->foods_eaten++;

        State new_food;
        do {
            new_food = get_random_food_loc(env->cols, env->rows);
        } while (
            new_food.x == env->snake.x &&
            new_food.y == env->snake.y
        );

        env->food = new_food;
    }

    if (
        env->snake.x < 0 ||
        env->snake.x >= (i32)env->cols ||
        env->snake.y < 0 ||
        env->snake.y >= (i32)env->rows
    ) {
        reward -= 10.0f;
    }

    return reward;
}

void reset_state(SnakeENV* env) {
    env->snake = (State){
        .x = (i32)(env->cols / 2),
        .y = (i32)(env->rows / 2),
    };

    do {
        env->food = get_random_food_loc(env->cols, env->rows);
    } while (
        env->food.x == env->snake.x &&
        env->food.y == env->snake.y
    );

    env->score = 0.0f;
    env->foods_eaten = 0;
    env->pov = RIGHT;
    env->steps = 0;
}

void take_action(SnakeENV* env, ACTION action) {
    if (action == NONE) {
        action = env->pov;
    }

    if(action == LEFT) {
        env->snake.x -= 1;
    }
    else if(action == RIGHT) {
        env->snake.x += 1;
    }
    else if(action == UP) {
        env->snake.y += 1;
    }
    else if(action == DOWN) {
        env->snake.y -= 1;
    }
    else {
        return;
    }

    env->pov = action;
}

ACTION sample_action(matrix* probs) {
    f32 r = prng_randf();
    f32 cdf = 0.0f;

    u32 size = probs->rows * probs->cols;
    for(u32 i=0; i<size; i++) {
        cdf += probs->data[i];
        if(r <= cdf) {
            return (ACTION)i;
        }
    }
    return (ACTION)(size - 1);
}


void build_state_vector(
    matrix* in,
    State state,
    State food_state,
    ACTION pov,
    u32 cols,
    u32 grid_size
) {
    clear(in);

    u32 snake_i = (u32)state.y * cols + (u32)state.x;
    u32 food_i = (u32)food_state.y * cols + (u32)food_state.x;

    in->data[snake_i] = 1.0f;
    in->data[grid_size + food_i] = 1.0f;
    in->data[2 * grid_size + (u32)pov] = 1.0f;
}

void train(
    model_state* model,
    SnakeENV* env,
    mem_arena* arena
) {
    (void)arena;

    u32 EPOCHS = 2000;
    u32 rollout_size = 64;
    u32 episode_len = 100;
    f32 gamma = 0.99f;
    f32 learning_rate = 0.05f;
    ReplayBuffer buffer = {0};

    for (u32 epoch=0; epoch < EPOCHS; epoch++) {
        u32 foods_eaten = 0;

        for(u32 i = 0; i < rollout_size; i++) {
            reset_state(env);

            Trajactory* traj = &buffer.trajactories[i];
            traj->len = 0;

            // Rollout phase - collect experience from the model
            for(u32 t = 0; t < episode_len; t++) {
                State state = env->snake;
                State food_state = env->food;
                ACTION pov = env->pov;

                build_state_vector(
                    model->input->val,
                    state,
                    food_state,
                    pov,
                    env->cols,
                    env->grid_size
                );

                forward_pass(&model->forward_graph);
                ACTION action = sample_action(model->output->val);

                take_action(env, action);

                // printf("X: %i\n", env->snake.x);
                f32 reward = get_reward(env);
                env->score += reward;
                b32 done = game_over(env);

                State next_state = env->snake;

                traj->states[t] = state;
                traj->food_states[t] = food_state;
                traj->povs[t] = pov;
                traj->actions[t] = action;
                traj->rewards[t] = reward;
                traj->next_states[t] = next_state;
                traj->dones[t] = done;
                traj->len = t + 1;

                if (done) {
                    break;
                }
            }

            foods_eaten += env->foods_eaten;
        }

        buffer.count = rollout_size;

        u32 sample_count = 0;
        f32 average_return = 0.0f;
        f32 return_sum = 0.0f;
        f32 return_sq_sum = 0.0f;

        // Compute reward-to-go statistics for a rollout-wide baseline.
        for (u32 b = 0; b < buffer.count; b++) {
            Trajactory* traj = &buffer.trajactories[b];
            f32 G = 0.0f;

            for (i32 t = (i32)traj->len - 1; t >= 0; t--) {
                G = traj->rewards[t] + gamma * G;
                traj->returns[t] = G;
                return_sum += G;
                return_sq_sum += G * G;
                sample_count++;
            }

            average_return += traj->returns[0];
            // printf("Returns %f\n", traj->returns[0]);  
        }
        
        

        


        f32 return_mean = return_sum / (f32)sample_count;
        f32 return_variance =
            return_sq_sum / (f32)sample_count - return_mean * return_mean;
        f32 return_std = sqrtf(MAX(return_variance, 0.0f));

        for (u32 i = 0; i < model->cost_graph.size; i++) {
            Var* cur = model->cost_graph.vars[i];

            if (cur->flags & VAR_FLAG_PARAMETER) {
                clear(cur->grad);
            }
        }

        // Training phase: use each trajectory from the current policy once.
        for (u32 b = 0; b < buffer.count; b++) {
            Trajactory* traj = &buffer.trajactories[b];

            for (u32 t = 0; t < traj->len; t++) {
                build_state_vector(
                    model->input->val,
                    traj->states[t],
                    traj->food_states[t],
                    traj->povs[t],
                    env->cols,
                    env->grid_size
                );

                clear(model->advantage->val);
                f32 advantage =
                    (traj->returns[t] - return_mean) / (return_std + 1e-8f);
                model->advantage->val->data[traj->actions[t]] = advantage;

                forward_pass(&model->cost_graph);
                backward_pass(&model->cost_graph);
            }
        }

        if (sample_count > 0) {
            f32 gradient_scale = learning_rate / (f32)sample_count;

            for (u32 i = 0; i < model->cost_graph.size; i++) {
                Var* cur = model->cost_graph.vars[i];

                if ((cur->flags & VAR_FLAG_PARAMETER) == 0) {
                    continue;
                }

                scale(cur->grad, gradient_scale);
                sub(cur->val, cur->val, cur->grad);
            }
        }

        printf(
            "Epoch %u | Average return: %.3f | Foods: %u | Samples: %u | Return std: %.3f\n",
            epoch, average_return / (f32)buffer.count,
            foods_eaten, sample_count, return_std
        );

    }
}


int main(void) {

    mem_arena* arena = arena_create(GiB(1));

    model_state* model = PUSH_STRUCT(arena, model_state);

    create_actor_model(arena, model);
    model->forward_graph = build_graph(arena, model, model->output);
    model->cost_graph = build_graph(arena, model, model->cost);


    SnakeENV* env = create_env(36);
    
    train(model, env, arena);

    return 0;
}
