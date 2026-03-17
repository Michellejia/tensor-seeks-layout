# Evaluation Metrics

The evaluation framework measures the effectiveness of layout selection algorithms across multiple dimensions: solution quality, compilation efficiency, and scalability. These metrics enable systematic comparison between exact solvers (treewidth DP, MaxSAT) and heuristic approaches (greedy search).

## Primary Metrics

### 1. Estimated Execution Cost

- **Definition:** Total cost computed by the cost model for executing the compiled program.
- **Units:** Abstract cost units (proportional to hardware cycles)
- **Purpose:** Primary optimization objective for layout selection algorithms

**Computation:**

```
Total Cost = Σ (Operator Execution Costs) + Σ (Transpose Costs)
```

**Components:**

- Operator costs: Sum of execution costs for all operations in the dataflow graph
- Transpose costs: Sum of layout conversion costs for all tensor transposes

**Interpretation:**

- Lower cost indicates better layout choices
- Reflects estimated runtime performance on target hardware
- Depends on cost model accuracy

### 2. Solution Quality (Relative Improvement)

- **Definition:** Percentage improvement over baseline heuristic.
- **Baseline:** Greedy heuristic (deterministic mode)

**Formula:**

```
Improvement (%) = (Baseline_Cost - Algorithm_Cost) / Baseline_Cost × 100%
```

**Interpretation:**

- Positive values indicate improvement over baseline
- Negative values indicate degradation
- Measures effectiveness of algorithmic approach

**Example:**

- Baseline cost: 1000 units
- MaxSAT cost: 850 units
- Improvement: (1000 - 850) / 1000 × 100% = 15%

### 3. Compilation Time

- **Definition:** Total time to compile the model, including layout selection and code generation.
- **Measurement:** Wall-clock time in seconds
- **Purpose:** Assess practical viability for production compilation
- **Trade-off:** Exact solvers may achieve better solution quality but require longer compilation time

**Components:**

- Layout selection algorithm runtime
- Layout verification (if enabled)
- Downstream compilation stages (code generation, optimization)

### 4. Layout Selection Time

- **Definition:** Time spent in layout selection algorithm only (excluding downstream compilation).
- **Measurement:** Wall-clock time in seconds
- **Purpose:** Isolate algorithmic efficiency from overall compilation pipeline

**Comparison:**

| Algorithm | Typical Time |
|-----------|-------------|
| Greedy heuristic | < 5 seconds |
| MaxSAT solver | 10–60 seconds (may timeout) |
| Treewidth DP | Depends on treewidth (fast for tw ≤ 3) |

## Secondary Metrics

### 5. Number of Transposes

- **Definition:** Count of layout conversion operations inserted by the compiler.
- **Purpose:** Measure data movement overhead
- **Interpretation:**
  - Fewer transposes generally indicate better layout propagation
  - Zero-cost transposes (reinterpretations) are not penalized
  - High transpose count may indicate suboptimal layout choices

### 6. Success Rate

- **Definition:** Percentage of models compiled successfully without errors.
- **Purpose:** Assess algorithm robustness and coverage

**Formula:**

```
Success Rate (%) = (Successful Compilations / Total Attempts) × 100%
```

**Failure Modes:**

- Solver timeout (MaxSAT)
- Memory exhaustion (large graphs)
- Layout verification failures

### 7. Cost Model Accuracy

- **Definition:** Correlation between estimated costs and actual hardware measurements.
- **Measurement:** Pearson correlation coefficient or mean absolute percentage error (MAPE)
- **Purpose:** Validate cost model fidelity
- **Note:** Requires hardware profiling data for validation

## Aggregate Statistics

### Geometric Mean

- **Definition:** Geometric mean of relative improvements across all models.
- **Purpose:** Summarize overall performance across diverse workloads
- **Advantage:** Less sensitive to outliers than arithmetic mean

**Formula:**

```
Geometric Mean = (∏ (1 + improvement_i))^(1/n) - 1
```

### Percentile Analysis

- **Median (50th percentile):** Typical performance
- **90th percentile:** Near-worst-case performance
- **Best case:** Maximum improvement observed
- **Purpose:** Understand performance distribution and variability

## Comparison Framework

### Baseline Normalization

All costs are normalized relative to the greedy heuristic baseline:

```
Normalized Cost = Algorithm_Cost / Baseline_Cost
```

**Interpretation:**

- `1.0` = Same as baseline
- `< 1.0` = Better than baseline
- `> 1.0` = Worse than baseline

### Statistical Significance

- **Method:** Paired t-test for comparing algorithms on the same models
- **Null Hypothesis:** No difference in mean costs between algorithms
- **Significance Level:** α = 0.05
- **Purpose:** Determine if observed improvements are statistically significant

## Model-Specific Analysis

### Grouping by Architecture

Models are grouped by architecture type for targeted analysis:

- Encoder-only transformers (BERT, RoBERTa)
- Encoder-decoder models (BART, Whisper)
- Decoder-only LLMs (GPT-style, OLMo)
- Mixture-of-experts (OLMoE)
- Multimodal models (Qwen-VL, IDEFICS)
- Convolutional architectures (ResNet, U-Net)
- Diffusion models (Stable Diffusion)

**Purpose:** Identify architecture-specific patterns in layout selection effectiveness

### Scalability Analysis

**Metrics by Model Size:**

- Small models (< 1B parameters)
- Medium models (1B–10B parameters)
- Large models (> 10B parameters)

**Purpose:** Assess how algorithms scale with model complexity

## Reporting Format

### Summary Table

| Algorithm | Avg. Improvement | Median Time (s) |
|-----------|-----------------|-----------------|
| Greedy (baseline) | 0% | 2.5 |
| Greedy Randomized | +2.3% | 2.8 |
| MaxSAT | +8.7% | 35.2 |
| Treewidth DP | +12.1% | 8.4 |

### Detailed Results

For each model:

- Model name and configuration
- Baseline cost
- Algorithm cost
- Relative improvement
- Compilation time
- Number of transposes
- Success/failure status

## Validation and Reproducibility

### Cost Model Validation

- **Method:** Compare estimated costs with actual hardware measurements for representative models
- **Metrics:**
  - Correlation coefficient (target: > 0.85)
  - Mean absolute percentage error (target: < 15%)
- **Purpose:** Ensure cost model provides reliable optimization signal

### Reproducibility Checks

**Requirements:**

- Fixed random seeds for deterministic algorithms
- Multiple runs for randomized algorithms (3–5 runs)
- Consistent compiler settings across experiments
- Documented hardware and software environment

## Practical Interpretation: When to Use Each Algorithm

| Algorithm | Best For |
|-----------|----------|
| Greedy Heuristic | Fast compilation (< 5s), production deployment, large-scale models |
| Greedy Randomized | Testing robustness, exploring near-optimal solutions, iterative optimization |
| MaxSAT Solver | Critical applications, medium-sized models (< 500 tensors), flexible compilation time |
| Treewidth DP | Low treewidth graphs (tw ≤ 3), optimality guarantees, predictable compilation time |
