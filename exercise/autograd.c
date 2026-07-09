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
static void var_cross_entropy_forward(Var* var);
static void var_cross_entropy_backward(Var* var);
static b32 var_shape_same(Var* a, Var* b, u32* rows, u32* cols);
static b32 var_shape_matmul(Var* a, Var* b, u32* rows, u32* cols);

static const VarType VAR_TYPE_CREATE = {
    .op = VAR_OP_CREATE,
    .num_inputs = 0,
};

const VarType VAR_TYPE_RELU = {
    .op = VAR_OP_RELU,
    .num_inputs = 1,
    .shape = var_shape_same,
    .forward = var_relu_forward,
    .backward = var_relu_backward,
};

const VarType VAR_TYPE_SOFTMAX = {
    .op = VAR_OP_SOFTMAX,
    .num_inputs = 1,
    .shape = var_shape_same,
    .forward = var_softmax_forward,
    .backward = var_softmax_backward,
};

const VarType VAR_TYPE_ADD = {
    .op = VAR_OP_ADD,
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_add_forward,
    .backward = var_add_backward,
};

const VarType VAR_TYPE_SUB = {
    .op = VAR_OP_SUB,
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_sub_forward,
    .backward = var_sub_backward,
};

const VarType VAR_TYPE_MATMUL = {
    .op = VAR_OP_MATMUL,
    .num_inputs = 2,
    .shape = var_shape_matmul,
    .forward = var_matmul_forward,
    .backward = var_matmul_backward,
};

const VarType VAR_TYPE_CROSS_ENTROPY = {
    .op = VAR_OP_CROSS_ENTROPY,
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_cross_entropy_forward,
    .backward = var_cross_entropy_backward,
};

Var* var_create(
    mem_arena* arena, model_state* model,
    u32 rows, u32 cols, u32 flags
) {
    Var* out = PUSH_STRUCT(arena, Var);
    if (out == NULL) {
        return NULL;
    }

    out->index = model->num_vars++;
    out->flags = flags;
    out->type = &VAR_TYPE_CREATE;
    out->val = create_matrix(arena, rows, cols);
    if (out->val == NULL) {
        return NULL;
    }

    if (flags & VAR_FLAG_REQUIRES_GRAD) {
        out->grad = create_matrix(arena, rows, cols);
        if (out->grad == NULL) {
            return NULL;
        }
    }

    return out;
}

Var* create_node(
    mem_arena* arena, model_state* model,
    const VarType* type, Var* a, Var* b, u32 flags
) {
    if (type == NULL || type->shape == NULL) {
        return NULL;
    }

    if (
        (a != NULL && (a->flags & VAR_FLAG_REQUIRES_GRAD)) ||
        (b != NULL && (b->flags & VAR_FLAG_REQUIRES_GRAD))
    ) {
        flags |= VAR_FLAG_REQUIRES_GRAD;
    }

    u32 rows = 0;
    u32 cols = 0;
    if (!type->shape(a, b, &rows, &cols)) {
        return NULL;
    }

    Var* out = var_create(arena, model, rows, cols, flags);
    if (out == NULL) {
        return NULL;
    }

    out->type = type;
    out->inputs[0] = a;
    if (type->num_inputs > 1) {
        out->inputs[1] = b;
    }

    return out;
}

Var* var_relu(
    mem_arena* arena, model_state* model,
    Var* input, u32 flags
) {
    return create_node(arena, model, &VAR_TYPE_RELU, input, NULL, flags);
}

Var* var_softmax(
    mem_arena* arena, model_state* model,
    Var* input, u32 flags
) {
    return create_node(arena, model, &VAR_TYPE_SOFTMAX, input, NULL, flags);
}

Var* var_add(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
) {
    return create_node(arena, model, &VAR_TYPE_ADD, a, b, flags);
}

Var* var_sub(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
) {
    return create_node(arena, model, &VAR_TYPE_SUB, a, b, flags);
}

Var* var_matmul(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
) {
    return create_node(arena, model, &VAR_TYPE_MATMUL, a, b, flags);
}

Var* var_cross_entropy(
    mem_arena* arena, model_state* model,
    Var* p, Var* q, u32 flags
) {
    return create_node(arena, model, &VAR_TYPE_CROSS_ENTROPY, p, q, flags);
}

Graph build_graph(
    mem_arena* arena, model_state* model, Var* out_var
) {
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);

    b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);

    u32 stack_size = 0;
    u32 out_size = 0;
    Var** stack = PUSH_ARRAY(scratch.arena, Var*, model->num_vars);
    Var** out = PUSH_ARRAY(scratch.arena, Var*, model->num_vars);

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
                    for (u32 k = j; k < stack_size - 1; k++) {
                        stack[k] = stack[k + 1];
                    }
                    stack_size--;
                }
            }

            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }

    Graph graph = {
        .size = out_size,
        .vars = PUSH_ARRAY_NZ(arena, Var*, out_size),
    };

    memcpy(graph.vars, out, sizeof(Var*) * out_size);

    arena_scratch_release(scratch);

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

        if ((cur->flags & VAR_FLAG_REQUIRES_GRAD) != VAR_FLAG_REQUIRES_GRAD) {
            continue;
        }

        if (cur->flags & VAR_FLAG_PARAMETER) {
            continue;
        }

        clear(cur->grad);
    }

    fill(graph->vars[graph->size - 1]->grad, 1.0f);

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

model_state* model_create(mem_arena* arena) {
    return PUSH_STRUCT(arena, model_state);
}

static b32 var_requires_grad(Var* var) {
    return var != NULL && (var->flags & VAR_FLAG_REQUIRES_GRAD);
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
    mul(var->val, var->inputs[0]->val, var->inputs[1]->val, true, false, false);
}

static void var_matmul_backward(Var* var) {
    Var* a = var->inputs[0];
    Var* b = var->inputs[1];

    if (var_requires_grad(a)) {
        mul(a->grad, var->grad, b->val, false, false, true);
    }

    if (var_requires_grad(b)) {
        mul(b->grad, a->val, var->grad, false, true, false);
    }
}

static void var_cross_entropy_forward(Var* var) {
    cross_entropy(var->val, var->inputs[0]->val, var->inputs[1]->val);
}

static void var_cross_entropy_backward(Var* var) {
    Var* p = var->inputs[0];
    Var* q = var->inputs[1];

    cross_entropy_add_grad(
        var_requires_grad(p) ? p->grad : NULL,
        var_requires_grad(q) ? q->grad : NULL,
        p->val, q->val, var->grad
    );
}

int main(void) {
    mem_arena* arena = arena_create(10000);
    if (arena == NULL) {
        return 1;
    }

    model_state* model = model_create(arena);
    if (model == NULL) {
        arena_destroy(arena);
        return 1;
    }

    u32 flags = VAR_FLAG_REQUIRES_GRAD | VAR_FLAG_PARAMETER;
    Var* var_a = var_create(arena, model, 25, 25, flags);

    Graph graph = build_graph(arena, model, var_a);
    printf("graph size: %u\n", graph.size);

    if (var_a == NULL) {
        arena_destroy(arena);
        return 1;
    }

    printf("model: %p\n", (void*)model);
    printf("var flags: %u\n", var_a->flags);

    arena_destroy(arena);
    return 0;
}


