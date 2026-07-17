#include "autograd.h"

static void var_relu_forward(Var* var);
static void var_relu_backward(Var* var);
static void var_softmax_forward(Var* var);
static void var_softmax_backward(Var* var);
static void var_add_forward(Var* var);
static void var_add_backward(Var* var);
static void var_sub_forward(Var* var);
static void var_sub_backward(Var* var);
static void var_matmul_forward(Var* var);
static void var_matmul_backward(Var* var);
static void var_reinforce_forward(Var* var);
static void var_reinforce_backward(Var* var);
static b32 var_shape_same(Var* a, Var* b, u32* rows, u32* cols);
static b32 var_shape_matmul(Var* a, Var* b, u32* rows, u32* cols);


const VarType VAR_TYPE_RELU = {
    .name = "relu",
    .num_inputs = 1,
    .shape = var_shape_same,
    .forward = var_relu_forward,
    .backward = var_relu_backward,
};

const VarType VAR_TYPE_SOFTMAX = {
    .name = "softmax",
    .num_inputs = 1,
    .shape = var_shape_same,
.forward = var_softmax_forward,
    .backward = var_softmax_backward,
};

const VarType VAR_TYPE_ADD = {
    .name = "add",
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_add_forward,
    .backward = var_add_backward,
};

const VarType VAR_TYPE_SUB = {
    .name = "sub",
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_sub_forward,
    .backward = var_sub_backward,
};

const VarType VAR_TYPE_MATMUL = {
    .name = "matmul",
    .num_inputs = 2,
    .shape = var_shape_matmul,
    .forward = var_matmul_forward,
    .backward = var_matmul_backward,
};

const VarType VAR_TYPE_REINFORCE_LOSS = {
    .name = "reinforce_loss",
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_reinforce_forward,
    .backward = var_reinforce_backward
};

Var* var_create(
    mem_arena* arena, model_state* model,
    u32 rows, u32 cols, u32 flags
) {
    Var* out = PUSH_STRUCT(arena, Var);

    out->index = model->num_vars++;
    out->flags = flags;
    out->type = NULL;
    out->val = create_matrix(arena, rows, cols);

    if (flags & VAR_FLAG_REQUIRES_GRAD) {
        out->grad = create_matrix(arena, rows, cols);
    }

    return out;
}

static b32 var_shape_same(Var* a, Var* b, u32* rows, u32* cols) {
    if (a == NULL) {
        return false;
    }

    if (b != NULL && (a->val->rows != b->val->rows || a->val->cols != b->val->cols)) {
        return false;
    }

    *rows = a->val->rows;
    *cols = a->val->cols;

    return true;
}

static b32 var_shape_matmul(Var* a, Var* b, u32* rows, u32* cols) {
    if (a == NULL || b == NULL) {
        return false;
    }

    if (a->val->cols != b->val->rows) {
        return false;
    }

    *rows = a->val->rows;
    *cols = b->val->cols;

    return true;
}



static Var* create_node(
    mem_arena* arena, model_state* model,
    const VarType* type, Var* a, Var* b
) {
    if (type == NULL || type->shape == NULL) {
        return NULL;
    }

    u32 flags = VAR_FLAG_NONE;
    if (
        (a != NULL && (a->flags & VAR_FLAG_REQUIRES_GRAD)) ||
        (b != NULL && (b->flags & VAR_FLAG_REQUIRES_GRAD))
    ) {
        flags = VAR_FLAG_REQUIRES_GRAD;
    }

    u32 rows = 0;
    u32 cols = 0;
    if (!type->shape(a, b, &rows, &cols)) {
        return NULL;
    }

    Var* out = var_create(arena, model, rows, cols, flags);

    out->type = type;
    out->inputs[0] = a;
    if (type->num_inputs > 1) {
        out->inputs[1] = b;
    }

    return out;
}

Var* var_relu(
    mem_arena* arena, model_state* model,
    Var* input
) {
    return create_node(arena, model, &VAR_TYPE_RELU, input, NULL);
}

Var* var_softmax(
    mem_arena* arena, model_state* model,
    Var* input
) {
    return create_node(arena, model, &VAR_TYPE_SOFTMAX, input, NULL);
}

Var* var_add(
    mem_arena* arena, model_state* model,
    Var* a, Var* b
) {
    return create_node(arena, model, &VAR_TYPE_ADD, a, b);
}

Var* var_sub(
    mem_arena* arena, model_state* model,
    Var* a, Var* b
) {
    return create_node(arena, model, &VAR_TYPE_SUB, a, b);
}

Var* var_matmul(
    mem_arena* arena, model_state* model,
    Var* a, Var* b
) {
    return create_node(arena, model, &VAR_TYPE_MATMUL, a, b);
}

Var* var_reinforce_loss(
    mem_arena* arena, model_state* model,
    Var* probs, Var* rt
) {
    return create_node(arena, model, &VAR_TYPE_REINFORCE_LOSS, probs, rt);
}


Graph build_graph(
    mem_arena* arena, model_state* model, Var* out_var
) {
    Graph graph = { 0 };
    if (arena == NULL || model == NULL || out_var == NULL || model->num_vars == 0) {
        return graph;
    }

    b8* visited = calloc(model->num_vars, sizeof(*visited));
    Var** stack = malloc(sizeof(*stack) * model->num_vars);
    Var** out = malloc(sizeof(*out) * model->num_vars);

    if (visited == NULL || stack == NULL || out == NULL) {
        free(visited);
        free(stack);
        free(out);
        return graph;
    }

    u32 stack_size = 0;
    u32 out_size = 0;

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        Var* cur = stack[--stack_size];

        if (cur->index >= model->num_vars) { continue; }

        if (visited[cur->index]) {
            if (out_size < model->num_vars) {
                out[out_size++] = cur;
            }
            continue;
        }
        
        visited[cur->index] = true;

        if (stack_size < model->num_vars) {
            stack[stack_size++] = cur;
        }

        u32 num_inputs = cur->type != NULL ? cur->type->num_inputs : 0;
        for (u32 i = 0; i < num_inputs; i++) {
            Var* input = cur->inputs[i];

            if (input->index >= model->num_vars || visited[input->index]) {
                continue;
            }

            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size-1; k++) {
                        stack[k] = stack[k+1];
                    }
                    stack_size--;
                }
            }

            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }


    graph.vars = PUSH_ARRAY_UNINIT(arena, Var*, out_size);
    if (graph.vars != NULL) {
        graph.size = out_size;
        memcpy(graph.vars, out, sizeof(*out) * out_size);
    }

    free(visited);
    free(stack);
    free(out);

    return graph;
}



void forward_pass(Graph* graph) {
    for (u32 i = 0; i < graph->size; i++) {
        Var* cur = graph->vars[i];

        if (cur->type != NULL && cur->type->forward != NULL) {
            cur->type->forward(cur);
        }
    }
}

void backward_pass(Graph* graph) {
    for (u32 i = 0; i < graph->size; i++) {
        Var* cur = graph->vars[i];

        if ((cur->flags & VAR_FLAG_REQUIRES_GRAD) == 0) {
            continue;
        }

        if (cur->flags & VAR_FLAG_PARAMETER) {
            continue;
        }

        clear(cur->grad);
    }

    fill(graph->vars[graph->size-1]->grad, 1.0f);

    for (i64 i = (i64)graph->size - 1; i >= 0; i--) {
        Var* cur = graph->vars[i];

        if ((cur->flags & VAR_FLAG_REQUIRES_GRAD) == 0) {
            continue;
        }

        if (cur->type != NULL && cur->type->backward != NULL) {
            cur->type->backward(cur);
        }
    }
}



static b32 var_requires_grad(Var* var) {
    return var != NULL && (var->flags & VAR_FLAG_REQUIRES_GRAD);
}

static void var_relu_forward(Var* var) {
    relu(var->val, var->inputs[0]->val);
}

static void var_relu_backward(Var* var) {
    Var* input = var->inputs[0];

    if (var_requires_grad(input)) {
        relu_add_grad(input->grad, input->val, var->grad);
    }
}

static void var_softmax_forward(Var* var) {
    softmax(var->val, var->inputs[0]->val);
}

static void var_softmax_backward(Var* var) {
    Var* input = var->inputs[0];

    if (var_requires_grad(input)) {
        softmax_add_grad(input->grad, var->val, var->grad);
    }
}

static void var_add_forward(Var* var) {
    add(var->val, var->inputs[0]->val, var->inputs[1]->val);
}

static void var_add_backward(Var* var) {
    Var* a = var->inputs[0];
    Var* b = var->inputs[1];

    if (var_requires_grad(a)) {
        add(a->grad, a->grad, var->grad);
    }

    if (var_requires_grad(b)) {
        add(b->grad, b->grad, var->grad);
    }
}

static void var_sub_forward(Var* var) {
    sub(var->val, var->inputs[0]->val, var->inputs[1]->val);
}

static void var_sub_backward(Var* var) {
    Var* a = var->inputs[0];
    Var* b = var->inputs[1];

    if (var_requires_grad(a)) {
        add(a->grad, a->grad, var->grad);
    }

    if (var_requires_grad(b)) {
        sub(b->grad, b->grad, var->grad);
    }
}

static void var_matmul_forward(Var* var) {
    matmul(var->val, var->inputs[0]->val, var->inputs[1]->val, 1, 0, 0);
}

static void var_matmul_backward(Var* var) {
    Var* a = var->inputs[0];
    Var* b = var->inputs[1];

    if (var_requires_grad(a)) {
        matmul(a->grad, var->grad, b->val, 0, 0, 1);
    }

    if (var_requires_grad(b)) {
        matmul(b->grad, a->val, var->grad, 0, 1, 0);
    }
}

static void var_reinforce_forward(Var* var) {
    reinforce_loss(var->val, var->inputs[0]->val, var->inputs[1]->val);
}

static void var_reinforce_backward(Var* var) {
    Var* probs = var->inputs[0];
    Var* rt = var->inputs[1];

    if (var_requires_grad(probs)) {
        reinforce_add_grad(probs->grad, probs->val, rt->val, var->grad);
    }
}

model_state* model_create(mem_arena* arena) {
    model_state* model = PUSH_STRUCT(arena, model_state);
    return model;
}
