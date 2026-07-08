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
