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
} ACTION;

typedef struct {
    i32 x;
    i32 y;
    u64 score;
    
    u32 rows;
    u32 cols;
    u32 grid_size;
    u32 food_pos_x;
    u32 food_pos_y;
    u64 steps;
    ACTION pov;
} SnakeENV;


#define BUFFER_SIZE 1024
#define EPISODE_LEN 100

typedef struct {
    i32 x;
    i32 y;
} State;

typedef struct {
    State states[EPISODE_LEN];
    ACTION actions[EPISODE_LEN];
    f32 rewards[EPISODE_LEN];
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

    env->x = 0;
    env->y = 0;
    env->rows = side;
    env->cols = side;
    env->grid_size = grid_size;
    env->score = 0;
    env->food_pos_x = 5;
    env->food_pos_y = 5;
    env->pov = RIGHT;
    env->steps = 0;
    return env;
}


b32 game_over(SnakeENV* env) {
    if (env->x > env->cols || env->x < 0) {
        return true;
    }
    else if(env->y > env->rows || env->y < 0) {
        return true;
    }
    return false;
}


f32 get_reward(const SnakeENV* env) {
    f32 reward = 0.0f;

    if (env->x == env->food_pos_x && env->y == env->food_pos_y) {
        reward += 5.0f;
    }

    if (env->x < 0 || env->x >= env->cols) {
        reward -= 1.0f;
    }

    if (env->y < 0 || env->y >= env->rows) {
        reward -= 1.0f;
    }

    return reward - 0.1;
}


void reset_state(SnakeENV* env) {

}

void take_action(SnakeENV* env, u32 action) {
    if (action == env->pov) { // 
        return;
    }
    else if(action == LEFT) {
        env->x -= 1;
    }
    else if(action == RIGHT) {
        env->x += 1;
    }
    else if(action == UP) {
        env->y += 1;
    }
    else if(action == DOWN) {
        env->y -= 1;
    }
}



void train(
    model_state* model,
    SnakeENV* env
) {
    u32 EPOCHS = 10;
    u32 batch_size = 32;
    u32 rollout_size = 128;
    u32 episode_len = 100;
    ReplayBuffer buffer = {0};

    for(u32 i = 0; i < rollout_size; i++) {
        // Rollout phase - collect experience from the model
        for(u32 t_i = 0; t_i < episode_len; t_i++) {
            // ACTION action = randn(4);
            forward_pass(&model->forward_graph);

            matrix* probs = model->output->val;
            f32 log_probs[probs->rows * probs->cols];
            for(u32 i=0; i<probs->rows * probs->cols; i++) {
                log_probs[i] = logf(probs->data[i]);
            } 

            ACTION action = argmax(model->output->val);
            

            printf("Output: %u\n", action);
            take_action(env, action);
            f32 reward = get_reward(env);
            env->score += reward;
            b32 is_game_over = game_over(env);

            // printf("Env %d\n",env->x);
            // printf("Env %d\n",env->y);
            printf("Env %llu\n",env->score);
            printf("GAME OVER: %u\n", is_game_over);     

            // Store the data in buffer
            buffer.trajactories[i].states[t_i].x = env->x;
            buffer.trajactories[i].states[t_i].y = env->y;
            buffer.trajactories[i].actions[t_i] = action;
            buffer.trajactories[i].rewards[t_i] = reward;
            buffer.trajactories[i].dones[t_i] = is_game_over;
            buffer.trajactories[i].len = t_i + 1;

            printf("x: %d\n", buffer.trajactories[i].states[t_i].x);

            if (is_game_over) {
                break;
            }
        }
    }
    // Training Phase
    for(u32 epoch=0; epoch < EPOCHS; epoch++) {
        
        // Sample batch from buffer
        u64 start_idx = randn( (u64)(BUFFER_SIZE - batch_size));
        u64 end_idx = start_idx + batch_size;

        for(u64 b=start_idx; b < end_idx; b++) {
            Trajactory traj = buffer.trajactories[b];
        }


    }


}


int main() {

    mem_arena* arena = arena_create(GiB(1));

    model_state* model = PUSH_STRUCT(arena, model_state);

    create_actor_critic_model(arena, model);
    model->forward_graph = build_graph(arena, model, model->output);
    model->cost_graph = build_graph(arena, model, model->cost);


    SnakeENV* env = create_env(36);

    // for(u32 i=0; i< 100; i++) {
    //     take_action(env, randn(4));
    //     f32 reward = get_reward(env);
    //     env->score += reward;
    //     b32 is_game_over = game_over(env);


    //     printf("Env %d\n",env->x);
    //     printf("Env %d\n",env->y);
    //     printf("Env %llu\n",env->score);
    //     printf("Env %f\n",reward);
    //     printf("GAME OVER: %u\n", is_game_over);

    //     if (is_game_over) {
    //         break;
    //     }
    // }
    train(model, env);

    return 0;
}
