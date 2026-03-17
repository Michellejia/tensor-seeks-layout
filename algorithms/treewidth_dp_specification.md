# Treewidth-Based Dynamic Programming

## Algorithm Overview

The treewidth-based algorithm provides an exact solution to the layout selection problem when the dataflow graph has bounded treewidth. It uses dynamic programming over a tree decomposition to compute optimal layout assignments.

## Theoretical Foundation

### Treewidth

**Definition:** The treewidth of a graph G = (V, E) is the minimum width over all tree decompositions of G, where the width of a tree decomposition is the size of its largest bag minus one.

**Key Property:** Graphs with bounded treewidth admit efficient dynamic programming algorithms for many NP-hard problems, including layout selection.

### Complexity

**Time Complexity:** O(|V| · |L|^(tw+1) · poly(|V|, |L|))

- |V|: number of tensors in the dataflow graph
- |L|: number of available layouts
- tw: treewidth of the dataflow graph
- poly(|V|, |L|): polynomial factor for computing operator and transpose costs

**Space Complexity:** O(|V| · |L|^tw)

**Practical Implication:** For graphs with small constant treewidth (e.g., tw ≤ 3), the algorithm runs in polynomial time.

## Algorithm Components

### 1. Tree Decomposition

**Input:** Dataflow graph G = (V, E)

**Output:** Tree decomposition T = (I, F) where:

- I is a collection of subsets of V (called "bags")
- F defines a tree structure over I
- Satisfies tree decomposition properties:
  - Every vertex v ∈ V appears in at least one bag
  - For every edge (u,v) ∈ E, there exists a bag containing both u and v
  - For every vertex v ∈ V, the bags containing v form a connected subtree

**Construction:** Use existing treewidth algorithms (e.g., from Bodlaender or Kloks) to compute a tree decomposition of width tw.

### 2. DP Table Structure

**DP State:** For each bag B_i in the tree decomposition and each partial layout assignment λ: B_i → L:

```
DP[i, λ] = minimum cost of optimally laying out the subgraph
           rooted at bag B_i, given that tensors in B_i
           have layouts specified by λ
```

**Table Size:** O(|I| · |L|^(tw+1)) entries, where |I| is the number of bags.

### 3. Inductive Computation

The algorithm processes bags in post-order (leaves to root):

#### Base Case (Leaf Bags)

For a leaf bag B_i with tensors {t_1, ..., t_k}:

```
DP[i, λ] = Σ_{op ∈ operators in B_i} cost_op(op, λ)
```

Where cost_op(op, λ) is the operator execution cost given layout assignment λ.

#### Recursive Case (Internal Bags)

For an internal bag B_i with child bags B_j1, ..., B_jm:

```
DP[i, λ] = min cost of operators in B_i
         + Σ_{j ∈ children} min_{λ' compatible with λ} (
             DP[j, λ'] + transpose_cost(λ, λ')
           )
```

Where:

- **Operator cost:** Sum of execution costs for operators whose tensors are all in B_i
- **Transpose cost:** Cost of layout conversions for tensors in B_i ∩ B_j that have different layouts in λ vs λ'
- **Compatibility:** λ' must agree with λ on tensors in B_i ∩ B_j (enforced by tree decomposition properties)

### 4. Solution Extraction

**Optimal Cost:** DP[root, λ\*] where λ\* minimizes DP[root, λ] over all possible assignments to the root bag.

**Optimal Assignment:** Backtrack through DP table to reconstruct the complete layout assignment for all tensors.

## Correctness

**Theorem:** The algorithm computes the optimal layout selection with minimum total cost.

**Proof Sketch:**

1. **Subproblem optimality:** Each DP state represents the optimal solution for a subgraph
2. **Overlapping subproblems:** Tree decomposition ensures each tensor's layout decision is captured in overlapping bags
3. **Optimal substructure:** Global optimum is composed of optimal solutions to subproblems
4. **Completeness:** Post-order traversal ensures all dependencies are resolved before computing parent bags

## Runtime Analysis

**Per-bag computation:**

- Enumerate |L|^|B_i| layout assignments for bag B_i
- For each assignment, compute operator costs: O(|ops| · cost_model_time)
- For each child, enumerate |L|^|B_i ∩ B_j| compatible assignments

**Total per bag:** O(|L|^(tw+1) · |children| · poly(|V|, |L|))

**Total runtime:** O(|V| · |L|^(tw+1) · poly(|V|, |L|))

## Practical Considerations

### When to Use This Algorithm

**Suitable for:**

- Dataflow graphs with small treewidth (tw ≤ 3)
- Critical subgraphs where optimality guarantees are required
- Validation of heuristic solutions

**Not suitable for:**

- Large graphs with high treewidth (exponential blowup)
- Real-time compilation scenarios requiring fast turnaround

### Implementation Notes

- **Treewidth computation:** Use heuristic tree decomposition algorithms for practical graphs
- **Cost model integration:** Plug in hardware-specific cost models for operators and transposes
- **Pruning:** Apply dominance pruning to reduce DP table size
- **Hybrid approach:** Combine with heuristics for high-treewidth subgraphs
