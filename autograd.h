#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "base.h"
#include "arena.h"
#include "mat.h"

typedef enum {
    VAR_FLAG_NONE = 0,
    VAR_FLAG_REQUIRES_GRAD  = (1 << 0),
    VAR_FLAG_PARAMETER      = (1 << 1),
} VarFlags;

typedef enum {
    VAR_OP_NULL = 0,
    VAR_OP_CREATE,

    VAR_OP_RELU,
    VAR_OP_SOFTMAX,
    VAR_OP_ADD,
    VAR_OP_SUB,
    VAR_OP_MATMUL,
    VAR_OP_CROSS_ENTROPY,
    VAR_OP_REINFORCE_LOSS,
} VarOp;

#define VAR_MAX_INPUTS 2

typedef struct Var Var;
typedef struct VarType VarType;
typedef void VarFunc(Var* var);
typedef b32 VarShapeFunc(Var* a, Var* b, u32* rows, u32* cols);

struct VarType {
    VarOp op;
    u32 num_inputs;
    VarShapeFunc* shape;
    VarFunc* forward;
    VarFunc* backward;
};

struct Var {
    u32 index;
    u32 flags;

    matrix* val;
    matrix* grad;

    const VarType* type;
    Var* inputs[VAR_MAX_INPUTS];
};

extern const VarType VAR_TYPE_RELU;
extern const VarType VAR_TYPE_SOFTMAX;
extern const VarType VAR_TYPE_ADD;
extern const VarType VAR_TYPE_SUB;
extern const VarType VAR_TYPE_MATMUL;
extern const VarType VAR_TYPE_CROSS_ENTROPY;
extern const VarType VAR_TYPE_REINFORCE_LOSS;

typedef struct {
    Var** vars;
    u32 size;
} Graph;

typedef struct {
    u32 num_vars;

    Var* input;
    Var* output;
    Var* desired_output;
    Var* cost;

    Graph forward_graph;
    Graph cost_graph;
} model_state;

Var* var_create(
    mem_arena* arena, model_state* model,
    u32 rows, u32 cols, u32 flags
);
Var* create_node(
    mem_arena* arena, model_state* model,
    const VarType* type, Var* a, Var* b, u32 flags
);

Var* var_relu(
    mem_arena* arena, model_state* model,
    Var* input, u32 flags
);
Var* var_softmax(
    mem_arena* arena, model_state* model,
    Var* input, u32 flags
);

Var* var_add(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
);
Var* var_sub(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
);
Var* var_matmul(
    mem_arena* arena, model_state* model,
    Var* a, Var* b, u32 flags
);
Var* var_cross_entropy(
    mem_arena* arena, model_state* model,
    Var* p, Var* q, u32 flags
);
Var* var_reinforce_loss(
    mem_arena* arena, model_state* model,
    Var* probs, Var* rt, u32 flags
);


Graph build_graph(
    mem_arena* arena, model_state* model, Var* out_var
);
void forward_pass(Graph* graph);
void backward_pass(Graph* graph);

model_state* model_create(mem_arena* arena);

#endif
