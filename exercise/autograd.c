#include "autograd.h"
#include "arena.h"

static void var_relu_forward(Var* var);
static void var_relu_backward(Var* var);
static void var_softmax_forward(Var* var);
static void var_softmax_backward(Var* var);
static void var_sub_forward(Var* var);
static void var_sub_backward(Var* var);
static void var_matmul_forward(Var* var);
static void var_matmul_backward(Var* var);
static void var_add_forward(Var* var);
static void var_add_backward(Var* var);
static void var_cross_entropy_forward(Var* var);
static void var_cross_entropy_backward(Var* var);
static b32 var_shape_same(Var* a, Var* b, u32* rows, u32* cols);
static b32 var_shape_matmul(Var* a, Var* b, u32* rows, u32* cols);

const VarType VAR_TYPE_RELU = {
    .op = VAR_OP_RELU,
    .num_inputs = 1,
    .shape = var_shape_same,
    .forward = var_relu_forward,
    .backward = var_relu_backward,
};

static const VarType VAR_TYPE_CREATE = {
    .op = VAR_OP_CREATE,
    .num_inputs = 0,
};

const VarType VAR_TYPE_SOFTMAX = {
    .op = VAR_OP_SOFTMAX,
    .num_inputs = 1,
    .shape = var_shape_same,
    .forward = var_softmax_forward,
    .backward = var_softmax_backward,
};

const VarType VAR_TYPE_CROSS_ENTROPY = {
    .op = VAR_OP_CROSS_ENTROPY,
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_cross_entropy_forward,
    .backward = var_cross_entropy_backward,
};

const VarType VAR_TYPE_MATMUL = {
    .op = VAR_OP_MATMUL,
    .num_inputs = 2,
    .shape = var_shape_matmul,
    .forward = var_matmul_forward,
    .backward = var_matmul_backward,
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
    if (type == NULL || type->shape == NULL){
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

    out->type = type;
    out->inputs[0] = a;
    if(type->num_inputs > 1) {
        out->inputs[1] = b;
    }
    return out;
}

Var* var_relu(
    mem_arena* arena, model_state* model,
    Var* input, u32 flags) {
    return create_node(
        arena, 
        model,
        &VAR_TYPE_RELU,
        input,
        NULL,
        flags
    );
}

Var* var_softmax(
    mem_arena* arena, model_state* model,
    Var* input, u32 flags
) {
    return create_node(
        arena,
        model,
        &VAR_TYPE_SOFTMAX,
        input,
        NULL,
        flags
    );
}

Var* var_add(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
) {
    return create_node(
        arena, 
        model,
        &VAR_TYPE_ADD,
        a, 
        b,
        flags
    );
}

Var* var_sub(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
) {
    return create_node(
        arena,
        model,
        &VAR_TYPE_SUB,
        a,
        b,
        flags
    );
}

Var* var_matmul(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
) {
    return create_node(
        arena,
        model,
        &VAR_TYPE_MATMUL,
        a,
        b,
        flags
    );
}

Var* var_cross_entropy(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
) {
    return create_node(
        arena,
        model,
        &VAR_TYPE_CROSS_ENTROPY,
        a,
        b,
        flags
    );
}


Graph build_graph(
    mem_arena* arena, 
    model_state* model,
    Var* out_var
) {
    mem_arena_temp scratch arena_scratch_get(&arena, 1);

    b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);
    
    u32 stack_size = 0;
    u32 out_size = 0;
    Var** stack = PUSH_ARRAY(scratch.arena, Var*, model->num_vars);
    Var** out = PUSH_ARRAY(scratch.arena, Var*, model->num_vars);

    stack[stack_size++] = out_var;

    while(stack_size > 0) {
        Var* cur = stack[--stack_size];

        if(cur->index >= model->num_vars) { continue; }
        if(visited[cur->index]) {
            if(out_size < model->num_vars) {
                out[out_size++] = cur;
            }
            continue;
        }
    }
    
    visited[cur->index] = true;

    if(stack_size < model->num_vars) {
        stack[stack_size++] = cur;
    }

    u32 num_inputs = cur->type != NULL ? cur->type->num_inputs : 0;
    for(u32 i=0; i < num_inputs; i++) {
        Var* input = cur->inputs[i];

        if(input->index >= model->num_vars || visited[input->index]) {
            continue;
        }

        for (u32 j=0; j < stack_size; j++) {
            if (stack[j] == input) {
                for (u32 k=j; j< stack_size-1; k++) {
                    stack[k] = stack[k+1];
                }
            }
        }

        if(stack_size < model->num_vars) {
            stack[stack_size++] = input;
        }
    }

    Graph graph = {
        .size = out_size,
        .vars = PUSH_ARRAY_NZ(arena, Var*, out_size);
    }

    memcpy(graph.vars, out, sizeof(Var*)* out_size);
    arena_scratch_release(scratch);
    
    return graph;
}

void forward_pass(Graph* graph) {
    for (u32 i=0; i<graph->size; i++) {
        Var* cur = graph->vars[i];

        if(cur->type != NULL & cur->type->forward != NULL) {
            cur->type->forward(cur);
        }
    }
}

void backward_pass(Graph* graph) {
    for(u32 i=0; i < graph->size; i++) {
        Var* cur = graph->vars[i];

        if((cur->flags & VAR_FLAG_REQUIRES_GRAD) != VAR_FLAG_REQUIRES_GRAD) {
            continue;
        }

        if(cur->flags & VAR_FLAG_PARAMETER) {
            continue;
        }

        clear(cur->grad);
    }

    fill(graph->vars[graph->size-1]->grad, 1.0f);

    for(i64 i= (i64)graph->size-1; i>=0; i--) {
        Var* cur-graph->vars[i];

        if ((cur->flags & VAR_FLAG_REQUIRES_GRAD) == 0) {
            continue;
        }
        if(cur->type != NULL && cur->type->backward != NULL) {
            cur->type->backward(cur);
        }
    }
}


model_state* model_create(mem_arena* arena) {
    model_state* model = PUSH_STRUCT(arena, model_state);

    return model;
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

static void var_relu_forward(Var* var) { (void)var; }
static void var_relu_backward(Var* var) { (void)var; }
static void var_softmax_forward(Var* var) { (void)var; }
static void var_softmax_backward(Var* var) { (void)var; }
static void var_sub_forward(Var* var) { (void)var; }
static void var_sub_backward(Var* var) { (void)var; }
static void var_matmul_forward(Var* var) { (void)var; }
static void var_matmul_backward(Var* var) { (void)var; }
static void var_add_forward(Var* var) { (void)var; }
static void var_add_backward(Var* var) { (void)var; }
static void var_cross_entropy_forward(Var* var) { (void)var; }
static void var_cross_entropy_backward(Var* var) { (void)var; }



int main() {

    mem_arena* arena = arena_create(10000);
    if (arena == NULL) {
        return 1;
    }

    matrix* a = create_matrix(arena, 25, 25);
    model_state* model = model_create(arena);
    if (model == NULL) {
        arena_destroy(arena);
        return 1;
    }

    printf("model: %p\n", (void*)model);

    u32 flags = VAR_FLAG_REQUIRES_GRAD | VAR_FLAG_PARAMETER;

    Var* var_a = var_create(
        arena, model, 25, 25, flags
    );
    if (var_a == NULL) {
        arena_destroy(arena);
        return 1;
    }
    printf("var ", var_a->val);

    printf("var flags: %u\n", var_a->flags);

    if (a == NULL) {
        arena_destroy(arena);
        return 1;
    }

    arena_destroy(arena);

    return 0;
}
