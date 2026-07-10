#define _CRT_SECURE_NO_WARNINGS


#include "base.h"
#include "arena.h"
#include "prng.h"
#include "mat.h"
#include "autograd.h"

#include "arena.c"
#include "prng.c"
#include "mat.c"
#include "autograd.c"

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} training_config;

void train(
    model_state* model,
    const training_config* config
);

void create_mnist_model(mem_arena* arena, model_state* model);

int main() {
    mem_arena* arena = arena_create(GiB(1));

    matrix* train_images = load_data(arena, 60000, 784, "train_images.mat");
    matrix* test_images = load_data(arena, 10000, 784, "test_images.mat");
    matrix* train_labels = create_matrix(arena, 60000, 10);
    matrix* test_labels = create_matrix(arena, 10000, 10);

    matrix* train_labels_file = load_data(arena, 60000, 1, "train_labels.mat");
    matrix* test_labels_file = load_data(arena, 10000, 1, "test_labels.mat");

    for(u32 i=0; i < 60000; i++) {
        u32 num = train_labels_file->data[i];
        train_labels->data[i * 10 + num] = 1.0f; 
    }

    for(u32 i=0; i < 10000; i++) {
        u32 num = test_labels_file->data[i];
        test_labels->data[i * 10 + num] = 1.0f;
    }

    model_state* model = model_create(arena);
    create_mnist_model(arena, model);
    model->forward_graph = build_graph(arena, model, model->output);
    model->cost_graph = build_graph(arena, model, model->cost);

    training_config config = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,
        .epochs = 10,
        .batch_size = 32,
        .learning_rate = 0.05
    };

    train(model, &config);

    return 0;
}


void create_mnist_model(mem_arena* arena, model_state* model) {
    Var* input = var_create(arena, model, 784, 1, VAR_FLAG_NONE);
    model->input = input;

    Var* W0 = var_create(arena, model, 16, 784, VAR_FLAG_REQUIRES_GRAD | VAR_FLAG_PARAMETER);
    Var* W1 = var_create(arena, model, 16, 16, VAR_FLAG_REQUIRES_GRAD | VAR_FLAG_PARAMETER);
    Var* W2 = var_create(arena, model, 10, 16, VAR_FLAG_REQUIRES_GRAD | VAR_FLAG_PARAMETER);

    f32 bound0 = sqrtf(6.0f / (784 + 16));
    f32 bound1 = sqrtf(6.0f / (16 + 16));
    f32 bound2 = sqrtf(6.0f / (16 + 10));
    fill_rand(W0->val, -bound0, bound0);
    fill_rand(W1->val, -bound1, bound1);
    fill_rand(W2->val, -bound2, bound2);

    Var* b0 = var_create(arena, model, 16, 1, VAR_FLAG_REQUIRES_GRAD | VAR_FLAG_PARAMETER);
    Var* b1 = var_create(arena, model, 16, 1, VAR_FLAG_REQUIRES_GRAD | VAR_FLAG_PARAMETER);
    Var* b2 = var_create(arena, model, 10, 1, VAR_FLAG_REQUIRES_GRAD | VAR_FLAG_PARAMETER);

    Var* z0_a = var_matmul(arena, model, W0, input, 0);
    Var* z0_b = var_add(arena, model, z0_a, b0, 0);

    Var* z1_a = var_matmul(arena, model, W1, z0_b, 0);
    Var* z1_b = var_add(arena, model, z1_a, b1, 0);

    Var* z2_a = var_matmul(arena, model, W2, z1_b, 0);
    Var* z2_b = var_add(arena, model, z2_a, b2, 0);

    Var* output = var_softmax(arena, model, z2_b, 0);
    model->output = output;

    Var* y = var_create(arena, model, 10, 1, 0);
    model->desired_output = y;

    Var* cost = var_cross_entropy(arena, model, y, output, 0);
    model->cost = cost;
}

void train(
    model_state* model,
    const training_config* config
) {
    matrix* train_images = config->train_images;
    matrix* train_labels = config->train_labels;
    matrix* test_images = config->test_images;
    matrix* test_labels = config->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;
    u32 num_tests = test_images->rows;

    u32 num_batches = num_examples / config->batch_size;

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);


    for (u32 epoch=0; epoch < config->epochs; epoch++) {

        printf("EPOCH %u\n", epoch);
        

        
        for (u32 batch=0; batch< num_batches; batch++) {
            
            f32 avg_loss = 0.0;

            for (u32 i=0; i < model->cost_graph.size; i++) {
                Var* cur = model->cost_graph.vars[i];


                if(cur->flags & VAR_FLAG_PARAMETER) {
                    clear(cur->grad);
                }

            }


            f32 avg_cost = 0.0f;
            for (u32 i=0; i < config->batch_size; i++) {
                u32 index = batch * config->batch_size + i;

                memcpy(
                    model->input->val->data,
                    train_images->data + index * input_size,
                    sizeof(f32) * input_size
                );

                memcpy(
                    model->desired_output->val->data,
                    train_labels->data + index * output_size,
                    sizeof(f32) * output_size
                );

                forward_pass(&model->cost_graph);
                backward_pass(&model->cost_graph);


                for (u64 i=0; i < model->cost->val->rows * model->cost->val->cols; i++) {
                    avg_loss += model->cost->val->data[i];
                } 
                


            }
            avg_loss = avg_loss / (f32)config->batch_size;

            for (u32 i=0; i < model->cost_graph.size; i++) {
                Var* cur = model->cost_graph.vars[i];
    
                if((cur->flags & VAR_FLAG_PARAMETER) != VAR_FLAG_PARAMETER) {
                    continue;
                }  
                
                matrix* grad = cur->grad;
                if (grad == NULL) {
                    continue;
                }

                f32 learning_rate = config->learning_rate / (f32)config->batch_size;
                u64 size = (u64)grad->rows * grad->cols;
                
    
                for(u64 i=0; i< size; i++) {
                    cur->val->data[i] -= learning_rate * grad->data[i];
                }            
            }

            printf("loss: %f\n", avg_loss);
        }
    }
}
