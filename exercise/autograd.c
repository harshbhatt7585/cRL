#include "autograd.h"

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
    Var* out  PUSH_STRUCT(arena, Var);

    out->index = model->num_vars++;
    out->flags = flags;
    out->type = &VAR_TYPE_CREATE;
    out->val = create_matrix(arena, rows, cols);

    if(flags & VAR_FLAG_REQUIRES_GRAD) {
        out->grad = create_matrix(arena, rows, cols);
    }



    return out;
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
    return 0;
}
