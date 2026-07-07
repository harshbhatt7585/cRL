
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
} model_training_desc;

void model_train(
    model_state* model,
    const model_training_desc* training_desc
);

void draw_mnist_digit(f32* data);
void print_prediction_snapshot(
    model_state* model,
    matrix* images, matrix* labels, u32 index,
    u32 epoch, u32 epochs, u32 batch, u32 num_batches,
    f32 avg_cost
);
void create_mnist_model(mem_arena* arena, model_state* model);

int main(void) {
    mem_arena* perm_arena = arena_create(GiB(1));

    matrix* train_images = load(perm_arena, 60000, 784, "train_images.mat");
    matrix* test_images = load(perm_arena, 10000, 784, "test_images.mat");
    matrix* train_labels = create(perm_arena, 60000, 10);
    matrix* test_labels = create(perm_arena, 10000, 10);

    {
        matrix* train_labels_file = load(perm_arena, 60000, 1, "train_labels.mat");
        matrix* test_labels_file = load(perm_arena, 10000, 1, "test_labels.mat");

        for (u32 i = 0; i < 60000; i++) {
            u32 num = train_labels_file->data[i];
            train_labels->data[i * 10 + num] = 1.0f;
        }

        for (u32 i = 0; i < 10000; i++) {
            u32 num = test_labels_file->data[i];
            test_labels->data[i * 10 + num] = 1.0f;
        }
    }

    draw_mnist_digit(test_images->data);
    for (u32 i = 0; i < 10; i++) {
        printf("%.0f ", test_labels->data[i]);
    }
    printf("\n\n");

    model_state* model = model_create(perm_arena);
    create_mnist_model(perm_arena, model);
    model->forward_graph = build_graph(perm_arena, model, model->output);
    model->cost_graph = build_graph(perm_arena, model, model->cost);

    memcpy(model->input->val->data, test_images->data, sizeof(f32) * 784);
    forward_pass(&model->forward_graph);

    printf("pre-training output: ");
    for (u32 i = 0; i < 10; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }
    printf("\n");

    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,

        .epochs = 10,
        .batch_size = 50,
        .learning_rate = 0.01f
    };
    model_train(model, &training_desc);
    
    memcpy(model->input->val->data, test_images->data, sizeof(f32) * 784);
    forward_pass(&model->forward_graph);
    printf("post-training output: ");
    for (u32 i = 0; i < 10; i++) {
        printf("%f ", model->output->val->data[i]);
    }
    printf("\n\n");


    arena_destroy(perm_arena);

    return 0;
}

void draw_mnist_digit(f32* data) {
    for (u32 y = 0; y < 28; y++) {
        for (u32 x = 0; x < 28; x++) {
            f32 num = data[x + y * 28];
            u32 col = 232 + (u32)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

void print_prediction_snapshot(
    model_state* model,
    matrix* images, matrix* labels, u32 index,
    u32 epoch, u32 epochs, u32 batch, u32 num_batches,
    f32 avg_cost
) {
    index %= images->rows;

    memcpy(
        model->input->val->data,
        images->data + index * images->cols,
        sizeof(f32) * images->cols
    );

    memcpy(
        model->desired_output->val->data,
        labels->data + index * labels->cols,
        sizeof(f32) * labels->cols
    );

    forward_pass(&model->cost_graph);

    u64 predicted = argmax(model->output->val);
    u64 expected = argmax(model->desired_output->val);

    printf("\x1b[H\x1b[J");
    printf(
        "Epoch %2u / %2u, Batch %4u / %4u, Average Cost: %.4f\n",
        epoch + 1, epochs, batch + 1, num_batches, avg_cost
    );
    printf(
        "Training image %u\n",
        index
    );
    printf(
        "MODEL PREDICTION: %llu, expected: %llu\n\n",
        (unsigned long long)predicted, (unsigned long long)expected
    );

    draw_mnist_digit(images->data + index * images->cols);

    printf("\n\noutput: ");
    for (u32 i = 0; i < model->output->val->rows * model->output->val->cols; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }
    printf("\n");
    fflush(stdout);
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
    Var* a0 = var_relu(arena, model, z0_b, 0);

    Var* z1_a = var_matmul(arena, model, W1, a0, 0);
    Var* z1_b = var_add(arena, model, z1_a, b1, 0);
    Var* z1_c = var_relu(arena, model, z1_b, 0);
    Var* a1 = var_add(arena, model, a0, z1_c, 0);

    Var* z2_a = var_matmul(arena, model, W2, a1, 0);
    Var* z2_b = var_add(arena, model, z2_a, b2, 0);
    Var* output = var_softmax(arena, model, z2_b, VAR_FLAG_NONE);
    model->output = output;

    Var* y = var_create(arena, model, 10, 1, VAR_FLAG_NONE);
    model->desired_output = y;

    Var* cost = var_cross_entropy(arena, model, y, output, VAR_FLAG_NONE);
    model->cost = cost;
}

void model_train(
    model_state* model,
    const model_training_desc* training_desc
) {
    matrix* train_images = training_desc->train_images;
    matrix* train_labels = training_desc->train_labels;
    matrix* test_images = training_desc->test_images;
    matrix* test_labels = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;
    u32 num_tests = test_images->rows;

    u32 num_batches = num_examples / training_desc->batch_size;

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    u32* training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
    for (u32 i = 0; i < num_examples; i++) {
        training_order[i] = i;
    }

    for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
        for (u32 i = 0; i < num_examples; i++) {
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;

            u32 tmp = training_order[b];
            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }

        for (u32 batch = 0; batch < num_batches; batch++) {
            for (u32 i = 0; i < model->cost_graph.size; i++) {
                Var* cur = model->cost_graph.vars[i];

                if (cur->flags & VAR_FLAG_PARAMETER) {
                    clear(cur->grad);
                }
            }

            f32 avg_cost = 0.0f;
            u32 snapshot_index = 0;
            for (u32 i = 0; i < training_desc->batch_size; i++) {
                u32 order_index = batch * training_desc->batch_size + i;
                u32 index = training_order[order_index];
                snapshot_index = index;

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

                avg_cost += sum(model->cost->val);
            }
            avg_cost /= (f32)training_desc->batch_size;

            for (u32 i = 0; i < model->cost_graph.size; i++) {
                Var* cur = model->cost_graph.vars[i];

                if ((cur->flags & VAR_FLAG_PARAMETER) != VAR_FLAG_PARAMETER) {
                    continue;
                }

                scale(
                    cur->grad,
                    training_desc->learning_rate /
                    training_desc->batch_size
                );
                sub(cur->val, cur->val, cur->grad);
            }

            print_prediction_snapshot(
                model,
                train_images, train_labels, snapshot_index,
                epoch, training_desc->epochs, batch, num_batches,
                avg_cost
            );
        }
        printf("\n");

        u32 num_correct = 0;
        f32 avg_cost = 0;
        for (u32 i = 0; i < num_tests; i++) {
            memcpy(
                model->input->val->data,
                test_images->data + i * input_size,
                sizeof(f32) * input_size
            );

            memcpy(
                model->desired_output->val->data,
                test_labels->data + i * output_size,
                sizeof(f32) * output_size
            );

            forward_pass(&model->cost_graph);

            avg_cost += sum(model->cost->val);
            num_correct +=
                argmax(model->output->val) ==
                argmax(model->desired_output->val);
        }

        avg_cost /= (f32)num_tests;
        printf(
            "Test Completed. Accuracy: %5d / %5d (%.1f%%), Average Cost: %.4f\n",
            num_correct, num_tests, (f32)num_correct / num_tests * 100.0f,
            avg_cost
        );
    }

    arena_scratch_release(scratch);
}
