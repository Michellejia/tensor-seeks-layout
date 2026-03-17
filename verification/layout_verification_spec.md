# Layout Verification Specification

A layout verification pass validates that memory layout annotations (partition axes, data layouts) are correctly applied to all operations in a dataflow graph after layout selection algorithms have executed.

**Goals:**

- Ensure all operations have valid layout annotations
- Verify layout choices satisfy hardware constraints
- Detect inconsistencies that could cause code generation failures
- Provide early error detection before expensive compilation stages

## Verification Framework

### Core Verification Principles

#### 1. Annotation Completeness

- **Principle:** Every operation in the dataflow graph must have layout information assigned.
- **Check:** Verify that no operation has null/undefined layout annotations.
- **Rationale:** Missing annotations indicate the layout selection algorithm failed to process certain operations, which will cause downstream compilation failures.

#### 2. Hardware Constraint Validation

- **Principle:** Layout choices must respect target hardware limitations.
- **Checks:**
  - Verify partition configurations don't exceed hardware parallelism limits
  - Ensure memory layout choices are supported by the target architecture
  - Validate that data movement patterns are feasible on the hardware
- **Example Constraint:** If hardware supports N-way parallelism, verify that the product of partition dimensions doesn't excessively exceed N.

#### 3. Operation-Specific Consistency

- **Principle:** Different operation types have specific layout requirements that must be satisfied.

**For Tensor Contractions** (e.g., matrix multiplication, convolution):

- **Requirement:** At least one of the following must be partitioned:
  - Reduction dimensions (enables parallel reduction)
  - Output dimensions (enables parallel output computation)
- **Rationale:** Without partitioning on either dimension, the operation is fully sequential, which may indicate a layout selection error

**Example:**

```
Matrix Multiplication: C[M,N] = A[M,K] × B[K,N]
- Reduction dimension: K
- Output dimensions: M, N
- Valid: Partition K (source partitioning)
- Valid: Partition M or N (dest partitioning)
- Invalid: No partitioning on any dimension
```

**For Memory Operations** (load/store):

- Verify layout annotations are consistent with tensor storage format
- Ensure partition choices enable efficient memory access patterns

## Verification Workflow

**Input:** Dataflow graph with layout annotations

**Output:** Validation report (errors, warnings)

```
For each operation in graph:
    1. Check annotation completeness
    2. Validate hardware constraints
    3. Verify operation-specific requirements
    4. Report errors or warnings
```

## Error Handling Strategy

### Error Severity Levels

**Errors** (compilation-blocking):

- Missing layout annotations
- Invalid tensor operation configurations
- Hardware constraint violations that prevent code generation

**Warnings** (non-blocking):

- Suboptimal but valid layout choices
- Configurations that may impact performance but are functionally correct

### Error Reporting

**For Errors:**

- Provide clear description of the violation
- Include operation identifier and context
- Terminate compilation to prevent downstream failures

**For Warnings:**

- Log the issue for developer awareness
- Allow compilation to continue
- Useful for identifying optimization opportunities

## Integration with Layout Selection

### Pipeline Position

The verification pass should be placed after layout selection but before code generation:

```
Layout Selection Pipeline:
1. Analyze layout requirements
2. Select layouts
3. → Verify layout annotations ← This pass
4. Insert layout conversions (transposes)
5. Generate code
```

### Validation Scope

The pass validates:

- Output of layout selection algorithms (treewidth DP, MaxSAT, greedy search)
- Consistency with hardware constraints
- Correctness of layout propagation through the graph

## Practical Usage Guidelines

### When to Enable Verification

**Enable for:**

- Development and testing of new layout algorithms
- Debugging layout-related compilation failures
- Validating correctness of layout selection implementations
- Regression testing after compiler changes

**Disable for:**

- Production compilation (after validation)
- Performance-critical scenarios
- Stable implementations with proven correctness

### Common Error Patterns

**Pattern 1: Missing Annotations**

- **Error:** Operation has no layout annotation
- **Cause:** Layout selection algorithm didn't cover this operation
- **Fix:** Extend algorithm coverage or add default layout assignment

**Pattern 2: Invalid Tensor Operation Partitioning**

- **Error:** Tensor contraction has no meaningful partitioning
- **Cause:** Neither reduction nor output dimensions are partitioned
- **Fix:** Adjust cost model to encourage partitioning on at least one dimension

**Pattern 3: Hardware Constraint Violation**

- **Warning:** Partition configuration exceeds hardware limits
- **Cause:** Over-aggressive partitioning
- **Fix:** Review partitioning strategy or adjust hardware constraints

## Implementation Considerations

### Extensibility

The verification framework should be extensible to support:

- New operation types with specific layout requirements
- Additional hardware constraints
- Custom validation rules for specialized architectures

### Performance

Verification should be lightweight:

- Linear time complexity in graph size
- Minimal memory overhead
- Fast enough for interactive development workflows

### Debugging Support

Provide detailed diagnostics:

- Operation identifiers for easy location in source
- Context about why a layout choice is invalid
- Suggestions for fixing common issues

## Relationship to Layout Selection Algorithms

This verification framework validates the output of layout selection algorithms described in Section 5 of the paper:

**Treewidth-based DP:**

- Verification ensures the optimal solution satisfies all constraints
- Catches implementation bugs in the DP algorithm

**MaxSAT encoding:**

- Verification validates that the SAT solver's solution is feasible
- Detects encoding errors or incomplete constraint specifications

**Greedy heuristics:**

- Verification catches cases where heuristics produce invalid layouts
- Provides safety net for approximate algorithms
