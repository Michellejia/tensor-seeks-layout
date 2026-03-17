# Greedy Randomized Search

The greedy randomized search algorithm (implemented as `PAGLayoutGreedySearchAlgorithm`) provides a practical heuristic for layout selection that balances solution quality with compilation time. It explores the layout space using sequential exhaustive search over partition axes groups (PAGs) while incorporating optional randomization to explore near-optimal solutions.

## Algorithm Components

### 1. Randomness Tolerance Parameter

- **Command-line option:** `--layout-randomness-tolerance`
- **Type:** Float (default: 0.0)
- **Semantics:**
  - `0.0` (deterministic): Selects the single best solution with minimum cost
  - `> 0.0` (randomized): Randomly selects from solutions within (1 + tolerance) × min_cost
  - Example: tolerance = 0.1 means solutions within 10% of optimal are candidates

**Usage:**

```python
--layout-randomness-tolerance=0.0  # Deterministic (production)
--layout-randomness-tolerance=0.1  # 10% tolerance (testing)
--layout-randomness-tolerance=0.3  # 30% tolerance (exploration)
```

### 2. Sequential Exhaustive Search

The algorithm processes dataflow subgraphs (DAGs) sequentially:

**Input:**

- Sorted ranges of partition axes groups (PAGs)
- Solution space (combinations of PAG assignments)
- Cost model for evaluating layouts

**Output:**

- Optimal (or near-optimal) PAG assignment
- Total cost
- Layout assignment for all tensors

**Algorithm Steps:**

1. **Initialize:** Set min_cost = ∞, near_optimal_solutions = []
2. **For each candidate solution:**
   - Compute total cost (operator costs + transpose costs)
   - If cost exceeds current minimum, prune (skip)
   - If deterministic mode (tolerance = 0.0):
     - Update best solution if cost < min_cost
   - If randomized mode (tolerance > 0.0):
     - If cost < min_cost: update min_cost, filter solutions outside tolerance
     - If cost ≤ min_cost × (1 + tolerance): add to near_optimal_solutions
3. **Select final solution:**
   - Deterministic: return best solution
   - Randomized: randomly select from near_optimal_solutions

### 3. Cost Computation

The algorithm computes three types of costs:

#### Operator Execution Cost

For each DAG with partition axes:

```
cost_dag = Σ_{inst in DAG} cost_model.inst_cost(inst, par_axes)
```

Implementation:

- Query hardware-specific cost model for each instruction
- Cache invalid layout checks to avoid redundant computation
- Aggregate costs across all instructions in the DAG

#### Tensor Contract Cost

For tensor contraction operations (e.g., matmul with reduction):

```
cost_tc = cost_model.inst_cost_tensor_contract(inst, src_par_axes, dst_par_axes)
```

Special handling:

- Instructions on contracted axes use source partition axes
- Instructions on non-contracted axes use destination partition axes
- Contraction operation itself uses specialized cost model

#### Transpose Cost

For state buffer tensors accessed with different layouts:

```
cost_transpose = Σ_{tensor} Σ_{access_layout ≠ tensor_layout}
                 estimate_transpose_cost(tensor, tensor_layout, access_layout)
```

Computation:

- Identify all load/store operations accessing the tensor
- Determine partition dimensions for each access
- Find optimal tensor layout minimizing total transpose cost
- Sum transpose costs for all mismatched accesses

### 4. Early Pruning

Solutions exceeding the current minimum cost are pruned immediately:

```python
if solution_cost > min_cost:
    # Skip this solution, don't add to candidates
    continue
```

**Benefit:** Reduces search time by avoiding evaluation of dominated solutions.

### 5. Dynamic Filtering

In randomized mode, the algorithm maintains a dynamically filtered set of near-optimal solutions:

```python
if new_solution_cost < min_cost:
    min_cost = new_solution_cost
    # Remove solutions now outside tolerance window
    near_optimal_solutions = [
        sol for sol in near_optimal_solutions
        if sol.cost <= min_cost * (1 + tolerance)
    ]
```

**Property:** The near-optimal set always contains solutions within tolerance of the current best.

## Complexity Analysis

### Time Complexity

- **Per solution evaluation:** O(|DAGs| × |insts_per_DAG| + |tensors| × |accesses_per_tensor|)
- **Total search:** O(|solution_space| × evaluation_time)

Where:

- |solution_space| = product of PAG choices per DAG
- Typically exponential in number of DAGs, but manageable for small subgraphs

### Space Complexity

- **Deterministic mode:** O(1) — only stores best solution
- **Randomized mode:** O(|near_optimal_solutions|) — stores all solutions within tolerance
- **Typical:** |near_optimal_solutions| ≪ |solution_space| due to cost filtering

## Performance Characteristics

### Deterministic Mode (tolerance = 0.0)

**Advantages:**

- Reproducible results across runs
- Minimal memory overhead
- Suitable for production compilation

**Limitations:**

- May miss alternative near-optimal solutions
- Sensitive to cost model inaccuracies

### Randomized Mode (tolerance > 0.0)

**Advantages:**

- Explores near-optimal solution space
- Provides diversity in layout choices
- Useful for testing robustness
- Can escape local optima in iterative compilation

**Limitations:**

- Non-deterministic results
- Higher memory usage
- Requires multiple runs for statistical analysis

## Integration with Cost Model

The algorithm integrates with `CycleBasedLayoutCostModel` which provides:

**Required methods:**

- `inst_cost(inst, dag, par_axes)` — Per-instruction execution cost
- `inst_cost_tensor_contract(inst, src_par_axes, dst_par_axes)` — Tensor contraction cost
- `invalid_layout_for_dag(dag, par_axes)` — Layout feasibility checking

**Cost model properties:**

- Hardware-specific (Trainium 1 vs Trainium 2)
- Accounts for memory bandwidth, compute throughput, and data movement
- Includes operator fusion opportunities

## Test Configuration

### Layout Algorithm Variants

- `"greedy"` → greedy search with tolerance = 0.0 (deterministic)
- `"smt"` → MaxSAT-based exact solver
- `"naive"` → baseline layout assignment

### Randomness Variants (for testing)

- `"greedy-rnd1"` → tolerance = 0.1 (10% tolerance)
- `"greedy-rnd2"` → tolerance = 0.3 (30% tolerance)

### Test Parameterization

```python
@pytest.mark.parametrize("custom_layout", ["greedy", "smt", "naive"])
def test_layout_selection(custom_layout):
    layout_algo, randomness = set_layout_and_randomness(custom_layout)
    # Run compilation with specified layout algorithm
```
