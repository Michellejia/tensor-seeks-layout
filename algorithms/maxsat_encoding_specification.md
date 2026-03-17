# MaxSAT Encoding

## Algorithm Overview

The MaxSAT encoding provides an exact solution to the layout selection problem by formulating it as a Maximum Satisfiability problem. Unlike the treewidth-based approach, this method works for general graphs without structural restrictions, making it more broadly applicable to real-world dataflow graphs.

## MaxSAT Problem Formulation

MaxSAT is an optimization variant of the Boolean satisfiability (SAT) problem where:

- **Hard clauses:** Must be satisfied (constraints)
- **Soft clauses:** Should be satisfied if possible (optimization objective)
- **Goal:** Find an assignment that satisfies all hard clauses while maximizing the number (or weighted sum) of satisfied soft clauses

## Encoding Scheme

### Variable Definitions

#### 1. Operator Variables

For each operator `op` in the dataflow graph and each feasible layout configuration `c`:

```
x_op,c : Boolean variable indicating whether operator op uses configuration c
```

**Semantics:** x_op,c = true iff operator `op` is executed with layout configuration `c`.

**Example:** For a matrix multiplication operator with input tensors A and B:

- `x_matmul,(row-major,row-major)` = true means both inputs use row-major layout
- `x_matmul,(col-major,row-major)` = true means A uses column-major, B uses row-major

#### 2. Transpose Variables

For each tensor `t` and each layout `ℓ`:

```
y_t,ℓ : Boolean variable indicating whether tensor t has layout ℓ
```

**Semantics:** y_t,ℓ = true iff tensor `t` is stored in memory with layout `ℓ`.

### Hard Constraints

These clauses enforce the validity of the layout assignment:

#### Constraint 1: Unique Operator Configuration

Each operator must use exactly one feasible configuration:

```
∀ op: (x_op,c1 ∨ x_op,c2 ∨ ... ∨ x_op,ck)  [at least one]
∀ op, ∀ c1 ≠ c2: (¬x_op,c1 ∨ ¬x_op,c2)      [at most one]
```

Encoding size: O(|ops| · |configs|²) clauses

#### Constraint 2: Unique Tensor Layout

Each tensor must have exactly one layout:

```
∀ t: (y_t,ℓ1 ∨ y_t,ℓ2 ∨ ... ∨ y_t,ℓm)  [at least one]
∀ t, ∀ ℓ1 ≠ ℓ2: (¬y_t,ℓ1 ∨ ¬y_t,ℓ2)      [at most one]
```

Encoding size: O(|tensors| · |layouts|²) clauses

#### Constraint 3: Operator-Tensor Consistency

If an operator uses a configuration, its input/output tensors must have compatible layouts:

```
∀ op, ∀ config c, ∀ tensor t used by op:
    x_op,c → y_t,layout_required_by(c,t)
```

Equivalently in CNF:

```
¬x_op,c ∨ y_t,ℓ  where ℓ = layout_required_by(c,t)
```

Encoding size: O(|ops| · |configs| · max_arity) clauses

### Soft Constraints (Objective Function)

These weighted clauses encode the optimization objective:

#### Soft Clause 1: Operator Execution Costs

For each operator `op` and configuration `c`:

- **Soft clause:** x_op,c with weight = cost_op(op, c)
- **Semantics:** Satisfying x_op,c incurs cost cost_op(op, c). The MaxSAT solver minimizes the total weighted cost.
- **Weight computation:** Use hardware-specific cost model to estimate execution time for operator `op` with configuration `c`.

#### Soft Clause 2: Transpose Costs

For each edge (t_producer, t_consumer) in the dataflow graph and each layout pair (ℓ1, ℓ2):

- **Soft clause:** (¬y_t,ℓ1 ∨ ¬needs_layout_ℓ2) with weight = transpose_cost(ℓ1, ℓ2)
- **Semantics:** If tensor `t` has layout ℓ1 but a consumer requires ℓ2, a transpose operation is needed with cost transpose_cost(ℓ1, ℓ2).
- **Optimization:** Transposes with zero cost (e.g., layout reinterpretation) can be omitted from the encoding.

## Complete Encoding

**Input:**

- Dataflow graph G = (V, E) with tensors V and dependencies E
- Set of layouts L
- Cost model: cost_op(op, config) and transpose_cost(ℓ1, ℓ2)

**Output:** MaxSAT instance (hard clauses, weighted soft clauses)

**Encoding:**

1. Create variables: x_op,c for all operators and feasible configs, y_t,ℓ for all tensors and layouts
2. Add hard clauses: unique operator config, unique tensor layout, operator-tensor consistency
3. Add soft clauses: operator costs (weight = cost_op), transpose costs (weight = transpose_cost)
4. Invoke MaxSAT solver (e.g., Z3, RC2, MaxHS)
5. Extract solution: layout assignment from satisfied y_t,ℓ variables

## Complexity Analysis

### Encoding Size

- **Variables:** O(|ops| · |configs| + |tensors| · |layouts|)
- **Hard clauses:** O(|ops| · |configs|² + |tensors| · |layouts|² + |ops| · |configs| · max_arity)
- **Soft clauses:** O(|ops| · |configs| + |edges| · |layouts|²)
- **Total encoding size:** O(|V| · |L|² + |E| · |L|²) where |V| = |tensors| + |ops|

### Solver Complexity

MaxSAT is NP-hard in general, but modern solvers (e.g., Z3, RC2) employ sophisticated techniques:

- **Core-guided algorithms:** Iteratively refine upper/lower bounds
- **SAT-based preprocessing:** Simplify formula before solving
- **Incremental solving:** Reuse learned clauses across iterations
- **Practical performance:** Depends on problem structure, but viable for graphs with hundreds of tensors and operators.

## Implementation Considerations

### 1. Cost Model Integration

**Operator costs:** Query hardware-specific cost model for each (operator, configuration) pair:

```python
cost_op(matmul, (row-major, row-major)) = estimate_matmul_cycles(...)
```

**Transpose costs:** Estimate data movement overhead:

```python
transpose_cost(row-major, col-major) = tensor_size * memory_bandwidth_cost
transpose_cost(layout, layout) = 0  # No-op reinterpretation
```

### 2. Solver Selection

**Recommended solvers:**

- **Z3:** General-purpose SMT solver with MaxSAT support (used in Mirage)
- **RC2:** Specialized MaxSAT solver with core-guided search
- **MaxHS:** Hybrid MaxSAT solver combining SAT and ILP techniques

**Configuration:** Use weighted partial MaxSAT mode with integer weights representing costs.

### 3. Scalability Optimizations

**Preprocessing:**

- Remove infeasible configurations early
- Merge equivalent layout choices
- Simplify transpose chains (e.g., A→B→C can be optimized to A→C)

**Incremental solving:**

- For large graphs, solve subgraphs independently and merge solutions
- Use solver assumptions to explore alternative solutions efficiently

**Timeout handling:**

- Set solver timeout (e.g., 60 seconds)
- Fall back to heuristic if solver times out
- Use partial solution as warm start for heuristic

### 4. Solution Extraction

From the satisfying assignment returned by the MaxSAT solver:

```python
def extract_layout_assignment(model):
    layout_assignment = {}
    for tensor in tensors:
        for layout in layouts:
            if model.evaluate(y[tensor, layout]):
                layout_assignment[tensor] = layout
                break
    return layout_assignment
```
