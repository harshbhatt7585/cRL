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
    State food_states[EPISODE_LEN];
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
    return env->x < 0 || env->x >= (i32)env->cols ||
           env->y < 0 || env->y >= (i32)env->rows;
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
    env->x = 0;
    env->y = 0;
    env->score = 0;
    env->food_pos_x = 5;
    env->food_pos_y = 5;
    env->pov = RIGHT;
    env->steps = 0;
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
    u32 cols,
    u32 grid_size
) {
    clear(in);

    u32 snake_i = (u32)state.y * cols + (u32)state.x;
    u32 food_i = (u32)food_state.y * cols + (u32)food_state.x;

    in->data[snake_i] = 1.0f;
    in->data[grid_size + food_i] = 1.0f;
}

void train(
    model_state* model,
    SnakeENV* env,
    mem_arena* arena
) {
    u32 EPOCHS = 512;
    u32 microbatch = 64;
    u32 batch_size = 32;
    u32 rollout_size = 128;
    u32 episode_len = 100;
    ReplayBuffer buffer = {0};

    for (u32 epoch=0; epoch < EPOCHS; epoch++) {

        for(u32 i = 0; i < rollout_size; i++) {
            reset_state(env);

            Trajactory* traj = &buffer.trajactories[i];
            traj->len = 0;

            // Rollout phase - collect experience from the model
            for(u32 t = 0; t < episode_len; t++) {
                State state = {
                    .x = env->x,
                    .y = env->y,
                };

                State food_state = {
                    .x = (i32)env->food_pos_x,
                    .y = (i32)env->food_pos_y,
                };

                build_state_vector(
                    model->input->val,
                    state,
                    food_state,
                    env->cols,
                    env->grid_size
                );

                forward_pass(&model->forward_graph);
                ACTION action = sample_action(model->output->val);

                take_action(env, action);

                f32 reward = get_reward(env);
                env->score += reward;
                b32 done = game_over(env);

                State next_state = {
                    .x = env->x,
                    .y = env->y,
                };

                traj->states[t] = state;
                traj->food_states[t] = food_state;
                traj->actions[t] = action;
                traj->rewards[t] = reward;
                traj->next_states[t] = next_state;
                traj->dones[t] = done;
                traj->len = t + 1;

                if (done) {
                    break;
                }
            }
        }
        // Training Phase
        for(u32 m=0; m < microbatch; m++) {
            
            // Sample batch from buffer
            u64 start_idx = randn( (u64)(BUFFER_SIZE - batch_size));
            u64 end_idx = start_idx + batch_size;
    
            for(u64 b=start_idx; b < end_idx; b++) {
                Trajactory traj = buffer.trajactories[b];
    
                matrix* policy_loss = create(arena, traj.len, 1); 
                f32 rt = 0.0f;
    
                for (u32 t=0; t < traj.len; t++) {
                    State state_ = traj.states[t];
                    State food_state_ = traj.food_states[t];
                    ACTION action_ = traj.actions[t];
                    f32 reward_ = traj.rewards[t];
                    f32 return_ = 0;

    
                    for (u32 j=t; j <traj.len; j++) {
                        return_ += traj.rewards[j];
                    }
                    if(t==(traj.len-1)) {
                        printf("Return: %f\n", return_);
                        rt = return_;
                    }
                    
                    build_state_vector(
                        model->input->val,
                        state_,
                        food_state_,
                        env->cols,
                        env->grid_size
                    );

                    clear(model->desired_output->val);
                    model->desired_output->val->data[action_] = return_;
                    
                    forward_pass(&model->cost_graph);
                    backward_pass(&model->cost_graph);
                    
    
                    f32 p = MAX(model->output->val->data[action_], 1e-8f);
                    policy_loss->data[t] = -logf(p) * return_;

                }
                // printf("Loss: %f\n", sum(policy_loss) / batch_size);

                for(u32 i=0; i< model->cost_graph.size; i++) {
                    Var* cur = model->cost_graph.vars[i];

                    if ((cur->flags & VAR_FLAG_PARAMETER) != VAR_FLAG_PARAMETER) {
                        continue;
                    }
                    scale(
                        cur->grad,
                        0.05f / batch_size 
                    );
                    sub(cur->val, cur->val, cur->grad);
                  }
                

                backward_pass(&model->cost_graph);
            }
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
    train(model, env, arena);

    return 0;
}
