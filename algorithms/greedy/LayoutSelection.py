"""
Greedy Layout Selection Algorithm

Exhaustive search over partition axes groups (PAGs) with early pruning
and optional randomization for near-optimal solution exploration.
"""

import itertools
import random
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass


# ============================================================================
# Data Structures
# ============================================================================

class Config:
    layout_randomness_tolerance: float = 0.0  # 0.0 = deterministic
    debug_mode: bool = False


@dataclass
class Node:
    id: str
    operations: List[str]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __repr__(self):
        return f"Node({self.id})"


@dataclass
class LayoutOption:
    id: int
    partition_axes: Set[int]
    cost_estimate: float = 0.0

    def __repr__(self):
        return f"Layout(axes={self.partition_axes}, cost={self.cost_estimate:.2f})"


@dataclass
class TensorContractInfo:
    """Describes a tensor contraction (e.g., matmul reduction) on an edge."""
    contracted_axes: Set[int]


@dataclass
class Edge:
    src: Node
    dst: Node
    contract: Optional[TensorContractInfo] = None


@dataclass
class Solution:
    node_to_layout: Dict[Node, LayoutOption]
    total_cost: float


# ============================================================================
# Cost Model
# ============================================================================

class CostModel:
    def __init__(self, base_compute_cost: float = 100.0,
                 transpose_penalty: float = 50.0,
                 memory_access_cost: float = 10.0,
                 contract_penalty: float = 75.0):
        self.base_compute_cost = base_compute_cost
        self.transpose_penalty = transpose_penalty
        self.memory_access_cost = memory_access_cost
        self.contract_penalty = contract_penalty

    def compute_node_cost(self, node: Node, layout: LayoutOption) -> float:
        cost = self.base_compute_cost * len(node.operations)
        cost += len(layout.partition_axes) * self.memory_access_cost
        return cost

    def compute_transpose_cost(self, src_layout: LayoutOption,
                               dst_layout: LayoutOption) -> float:
        if src_layout.partition_axes == dst_layout.partition_axes:
            return 0.0
        diff = src_layout.partition_axes.symmetric_difference(dst_layout.partition_axes)
        return len(diff) * self.transpose_penalty

    def compute_tensor_contract_cost(self, src_layout: LayoutOption,
                                     dst_layout: LayoutOption,
                                     contract: TensorContractInfo) -> float:
        """Cost for a tensor contraction edge.

        Instructions on contracted axes use the source partition;
        non-contracted axes use the destination partition.
        """
        src_on_contracted = src_layout.partition_axes & contract.contracted_axes
        dst_on_non_contracted = dst_layout.partition_axes - contract.contracted_axes
        cost = len(src_on_contracted) * self.contract_penalty
        cost += len(dst_on_non_contracted) * self.memory_access_cost
        # Mismatch penalty for axes that appear in both but differ
        overlap = (src_layout.partition_axes - contract.contracted_axes)
        diff = overlap.symmetric_difference(dst_on_non_contracted)
        cost += len(diff) * self.transpose_penalty
        return cost


# ============================================================================
# Greedy Search Algorithm
# ============================================================================

class GreedyLayoutSearchAlgorithm:
    """Exhaustive search with early pruning and optional randomness."""

    def __init__(self, cost_model: CostModel, config: Config = None):
        self.cost_model = cost_model
        self.config = config or Config()
        self.best_solution: Optional[Solution] = None

    def search(self, nodes: List[Node],
               node_to_layout_options: Dict[Node, List[LayoutOption]],
               edges: List[Edge]) -> Solution:
        sorted_nodes = sorted(nodes, key=lambda n: n.id)
        search_space = [(node, node_to_layout_options[node]) for node in sorted_nodes]
        layout_indices = [range(len(opts)) for _, opts in search_space]
        all_solutions = itertools.product(*layout_indices)

        best = self._sequential_exhaustive_search(search_space, all_solutions, edges)
        self.best_solution = best
        return best

    def _sequential_exhaustive_search(self, search_space, solutions, edges):
        min_cost = None
        min_cost_solution = None
        tol = self.config.layout_randomness_tolerance
        near_optimal = [] if tol > 0 else None
        count = 0
        pruned = 0

        for indices in solutions:
            count += 1
            mapping = {}
            for i, idx in enumerate(indices):
                mapping[search_space[i][0]] = search_space[i][1][idx]

            cost = self._evaluate(mapping, edges, min_cost)
            if cost is None:
                pruned += 1
                continue

            if self.config.debug_mode:
                print(f"Solution {count}: cost={cost:.2f}")

            if tol > 0:
                if min_cost is None or cost < min_cost:
                    min_cost = cost
                    near_optimal = [
                        s for s in near_optimal
                        if s.total_cost <= min_cost * (1 + tol)
                    ]
                    near_optimal.append(Solution(mapping.copy(), cost))
                elif cost <= min_cost * (1 + tol):
                    near_optimal.append(Solution(mapping.copy(), cost))
            else:
                if min_cost is None or cost < min_cost:
                    min_cost = cost
                    min_cost_solution = Solution(mapping.copy(), cost)

        if tol > 0 and near_optimal:
            min_cost_solution = random.choice(near_optimal)
            if self.config.debug_mode:
                print(f"Selected 1 of {len(near_optimal)} near-optimal "
                      f"solutions (tolerance={tol})")

        if self.config.debug_mode:
            print(f"Search complete: {count} evaluated, {pruned} pruned")

        return min_cost_solution

    def _evaluate(self, mapping, edges, current_min):
        total = 0.0
        for node, layout in mapping.items():
            total += self.cost_model.compute_node_cost(node, layout)
            if current_min is not None and total >= current_min:
                return None

        for edge in edges:
            src_l = mapping[edge.src]
            dst_l = mapping[edge.dst]
            if edge.contract is not None:
                total += self.cost_model.compute_tensor_contract_cost(
                    src_l, dst_l, edge.contract)
            else:
                total += self.cost_model.compute_transpose_cost(src_l, dst_l)
            if current_min is not None and total >= current_min:
                return None

        return total


# ============================================================================
# Demo
# ============================================================================

def run_example():
    nodes = [Node(id=f"node_{i}", operations=["matmul", "add"]) for i in range(4)]
    options = {}
    for n in nodes:
        options[n] = [
            LayoutOption(0, set()),
            LayoutOption(1, {0}),
            LayoutOption(2, {0, 1}),
        ]

    edges = [
        Edge(nodes[0], nodes[1]),
        Edge(nodes[1], nodes[2], TensorContractInfo(contracted_axes={1})),
        Edge(nodes[2], nodes[3]),
    ]

    cm = CostModel()

    print("=== Deterministic (tolerance=0.0) ===")
    cfg = Config()
    cfg.layout_randomness_tolerance = 0.0
    cfg.debug_mode = True
    sol = GreedyLayoutSearchAlgorithm(cm, cfg).search(nodes, options, edges)
    print(f"Best cost: {sol.total_cost:.2f}\n")

    print("=== Randomized (tolerance=0.1) ===")
    cfg2 = Config()
    cfg2.layout_randomness_tolerance = 0.1
    cfg2.debug_mode = True
    sol2 = GreedyLayoutSearchAlgorithm(cm, cfg2).search(nodes, options, edges)
    print(f"Selected cost: {sol2.total_cost:.2f}")


if __name__ == "__main__":
    run_example()
