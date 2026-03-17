# Cost Model Interface

The cost model is a critical component that estimates the execution cost of operations under different layout configurations. Layout selection algorithms (treewidth DP, MaxSAT, greedy search) query the cost model to evaluate candidate solutions and select layouts that minimize total execution cost.

**Purpose:**

- Estimate operator execution costs for different layout configurations
- Estimate transpose (layout conflict) costs between layouts
- Enable cost-driven optimization in layout selection algorithms

## Core Methods

### 1. Operator Cost Estimation

**Method:** `operator_cost(op, layout)`

- **Input:**
  - `op`: The operation to be executed (e.g., matrix multiplication, convolution, elementwise operation)
  - `layout`: Layout configuration specifying the memory layout for each input/output tensor
- **Output:** Estimated execution cost (typically in cycles or time units)
- **Purpose:** Estimates the cost of executing an operator with a specific layout configuration.

**Example:**

```
MatMul: C[M,N] = A[M,K] × B[K,N]

layout_1 = {A: row-major, B: row-major, C: row-major}
cost_1 = operator_cost(matmul, layout_1)

layout_2 = {A: row-major, B: col-major, C: row-major}
cost_2 = operator_cost(matmul, layout_2)
```

**Cost Factors:**

- Memory access patterns (sequential vs. strided)
- Cache utilization
- Compute throughput
- Data movement overhead
- Hardware-specific characteristics

### 2. Transpose Cost Estimation

**Method:** `transpose_cost(tensor, src_layout, dst_layout)`

- **Input:**
  - `tensor`: The tensor being transposed
  - `src_layout`: Current memory layout
  - `dst_layout`: Desired memory layout
- **Output:** Estimated cost of layout conversion (typically in cycles or time units)
- **Purpose:** Estimates the cost of converting a tensor from one layout to another.

**Example:**

```
tensor = A[M,K]
cost = transpose_cost(A, row-major, col-major)
```

**Cost Factors:**

- Tensor size (number of elements)
- Memory bandwidth
- Cache effects
- Whether conversion can be done in-place or requires additional memory

**Special Cases:**

- **Zero-cost transposes:** Layout reinterpretations that don't require data movement
- **Fused transposes:** Transposes that can be fused with adjacent operations

### 3. Layout Feasibility Check

**Method:** `is_feasible(op, layout)`

- **Input:**
  - `op`: The operation to check
  - `layout`: Candidate layout configuration
- **Output:** Boolean indicating whether the operator supports the given layout
- **Purpose:** Filters out invalid layout configurations before cost evaluation. Infeasible layouts are excluded from the search space or assigned infinite cost.

**Example:**

```
# Some operators may not support certain layouts
is_feasible(conv2d, {input: NHWC, filter: OIHW})  # True
is_feasible(conv2d, {input: NCHW, filter: OIHW})  # False on some hardware
```

## Hardware-Specific Cost Models

Cost models are typically hardware-specific and account for:

**Memory Hierarchy:**

- Register file access latency
- L1/L2 cache characteristics
- Main memory bandwidth
- DMA transfer costs

**Compute Resources:**

- Arithmetic unit throughput
- Parallelism (SIMD, vector units)
- Specialized accelerators (tensor cores, systolic arrays)

**Data Movement:**

- Memory access patterns (coalesced vs. scattered)
- Bank conflicts
- Prefetching opportunities

## Integration with Layout Selection Algorithms

### Treewidth-Based DP

The DP algorithm queries the cost model for each candidate layout assignment:

```
For each bag in tree decomposition:
    For each layout assignment λ:
        cost = 0
        For each operator in bag:
            cost += operator_cost(operator, λ)
        For each child bag:
            For each compatible assignment λ':
                cost += transpose_cost(tensors, λ, λ')
        DP[bag, λ] = min(cost, DP[bag, λ])
```

### MaxSAT Encoding

The MaxSAT encoding uses cost model estimates as weights for soft clauses:

```
For each operator and configuration:
    weight = operator_cost(operator, config)
    Add soft clause with weight

For each tensor and layout pair:
    weight = transpose_cost(tensor, layout1, layout2)
    Add soft clause with weight
```

### Greedy Randomized Search

The greedy algorithm queries the cost model to evaluate each candidate solution:

```
For each candidate solution:
    total_cost = 0
    For each operator:
        total_cost += operator_cost(operator, solution.layout)
    For each transpose:
        total_cost += transpose_cost(tensor, src_layout, dst_layout)
    If total_cost < best_cost:
        best_solution = solution
```

## Cost Model Accuracy

### Importance of Accuracy

**Impact on Layout Selection:**

- Accurate cost models lead to better layout choices
- Inaccurate models may cause suboptimal layouts to be selected
- Cost model errors compound across the dataflow graph

**Validation:**

- Compare estimated costs with actual hardware measurements
- Iteratively refine cost models based on profiling data
- Use machine learning to improve cost predictions

## Example Cost Model Workflow

```python
# Layout selection algorithm queries cost model

# Step 1: Estimate operator costs
matmul_cost_row_row = operator_cost(
    operator=matmul,
    layout_config={A: row-major, B: row-major}
)

matmul_cost_row_col = operator_cost(
    operator=matmul,
    layout_config={A: row-major, B: col-major}
)

# Step 2: Estimate transpose costs
transpose_cost = transpose_cost(
    tensor=B,
    source_layout=row-major,
    target_layout=col-major
)

# Step 3: Compare total costs
total_cost_1 = matmul_cost_row_row  # No transpose needed
total_cost_2 = matmul_cost_row_col + transpose_cost

# Step 4: Select layout with minimum cost
if total_cost_1 < total_cost_2:
    selected_layout = {A: row-major, B: row-major}
else:
    selected_layout = {A: row-major, B: col-major}
```
