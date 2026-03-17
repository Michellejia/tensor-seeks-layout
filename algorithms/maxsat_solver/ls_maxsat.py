#!/usr/bin/env python3
"""MaxSAT-based exact layout selection solver.

Encodes the layout selection problem as a weighted partial MaxSAT instance
and solves it using Z3's Optimize (pseudo-boolean) backend.

Input format: same dump format as the treewidth solver.
"""

import argparse
import sys
import time
from collections import defaultdict

try:
    from z3 import Bool, Optimize, And, Or, Not, If, Int, is_true, sat
except ImportError:
    sys.exit("z3-solver is required: pip install z3-solver")


# ---------------------------------------------------------------------------
# Parsing (mirrors the treewidth solver's dump format)
# ---------------------------------------------------------------------------

def parse_dump(path):
    sections = defaultdict(list)
    current = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# "):
                current = line[2:].strip()
                continue
            if current is None or line.strip() == "":
                continue
            sections[current].append(line)

    operator_costs = []  # (op, partition, cost)
    for row in sections["operator_costs"]:
        cols = [c.strip() for c in row.split(";")]
        operator_costs.append((cols[0], cols[1], int(cols[2].replace(",", ""))))

    tensor_layouts = []  # (op, partition, tensor, layout, role)
    for row in sections["tensor_layouts"]:
        cols = [c.strip() for c in row.split(";")]
        tensor_layouts.append((cols[0], cols[1], cols[2], cols[3], cols[4]))

    transpose_costs = []  # (tensor, src, dst, cost)
    for row in sections["tensor_transpose_costs"]:
        cols = [c.strip() for c in row.split(";")]
        transpose_costs.append((cols[0], cols[1], cols[2], int(cols[3].replace(",", ""))))

    edges = []
    for row in sections["edges"]:
        cols = [c.strip() for c in row.split(";")]
        edges.append((cols[0], cols[1]))

    return operator_costs, tensor_layouts, transpose_costs, edges


# ---------------------------------------------------------------------------
# MaxSAT encoding
# ---------------------------------------------------------------------------

def solve(path, timeout_s=600):
    operator_costs, tensor_layouts, transpose_costs, edges = parse_dump(path)

    # Collect operators and their configurations.
    op_configs = defaultdict(dict)  # op -> {partition: cost}
    for op, part, cost in operator_costs:
        op_configs[op][part] = cost

    # Collect tensors and their possible layouts.
    tensor_possible = defaultdict(set)
    choice_uses = defaultdict(list)  # (op, part) -> [(tensor, layout, role)]
    for op, part, tensor, layout, role in tensor_layouts:
        tensor_possible[tensor].add(layout)
        choice_uses[(op, part)].append((tensor, layout, role))

    opt = Optimize()
    opt.set("timeout", timeout_s * 1000)

    # --- Variables ---
    # x[op][part] = Bool: operator op uses partition part
    x = {}
    for op, parts in op_configs.items():
        x[op] = {part: Bool(f"x_{op}_{part}") for part in parts}

    # y[tensor][layout] = Bool: tensor has this layout
    y = {}
    for tensor, layouts in tensor_possible.items():
        y[tensor] = {l: Bool(f"y_{tensor}_{l}") for l in layouts}

    # --- Hard constraints ---

    # HC1: Each operator uses exactly one configuration.
    for op, parts in x.items():
        bvars = list(parts.values())
        opt.add(Or(*bvars))  # at-least-one
        for i in range(len(bvars)):
            for j in range(i + 1, len(bvars)):
                opt.add(Not(And(bvars[i], bvars[j])))  # at-most-one

    # HC2: Each tensor has exactly one layout.
    for tensor, layouts in y.items():
        bvars = list(layouts.values())
        opt.add(Or(*bvars))
        for i in range(len(bvars)):
            for j in range(i + 1, len(bvars)):
                opt.add(Not(And(bvars[i], bvars[j])))

    # HC3: Operator-tensor consistency.
    for (op, part), uses in choice_uses.items():
        if op not in x or part not in x[op]:
            continue
        for tensor, layout, _role in uses:
            if tensor in y and layout in y[tensor]:
                opt.add(Or(Not(x[op][part]), y[tensor][layout]))

    # --- Soft constraints (objective) ---
    total_cost = Int("total_cost")
    cost_expr = 0

    # Operator execution costs.
    for op, parts in op_configs.items():
        for part, cost in parts.items():
            cost_expr += If(x[op][part], cost, 0)

    # Transpose costs.
    for tensor, src, dst, cost in transpose_costs:
        if tensor in y and src in y[tensor] and dst in y[tensor]:
            # Cost incurred when tensor has src layout but needs dst.
            cost_expr += If(And(y[tensor][src], Not(y[tensor][dst])), cost, 0)

    opt.add(total_cost == cost_expr)
    opt.minimize(total_cost)

    # --- Solve ---
    t0 = time.time()
    result = opt.check()
    elapsed = time.time() - t0

    if result != sat:
        print(f"Solver returned: {result} ({elapsed:.3f}s)", file=sys.stderr)
        return None

    model = opt.model()
    obj = model[total_cost].as_long()

    # Extract assignments.
    op_assignment = {}
    for op, parts in x.items():
        for part, var in parts.items():
            if is_true(model[var]):
                op_assignment[op] = part
                break

    tensor_assignment = {}
    for tensor, layouts in y.items():
        for layout, var in layouts.items():
            if is_true(model[var]):
                tensor_assignment[tensor] = layout
                break

    return {
        "objective": obj,
        "elapsed_s": elapsed,
        "op_assignment": op_assignment,
        "tensor_assignment": tensor_assignment,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_result(res):
    if res is None:
        print("UNSATISFIABLE or TIMEOUT")
        return
    print(f"objective: {res['objective']}")
    print(f"runtime_s: {res['elapsed_s']:.3f}")
    print()
    print("# operator_selection")
    for op in sorted(res["op_assignment"]):
        print(f"{op};{res['op_assignment'][op]}")
    print()
    print("# tensor_selection")
    for t in sorted(res["tensor_assignment"]):
        print(f"{t};{res['tensor_assignment'][t]}")


def main():
    ap = argparse.ArgumentParser(description="MaxSAT layout selection solver")
    ap.add_argument("input", help="Path to dump file")
    ap.add_argument("--timeout", type=int, default=600, help="Solver timeout in seconds")
    args = ap.parse_args()

    res = solve(args.input, timeout_s=args.timeout)
    print_result(res)


if __name__ == "__main__":
    main()
