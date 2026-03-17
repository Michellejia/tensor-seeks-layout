# MaxSAT Layout Selection Solver

A Python implementation of the MaxSAT-based exact layout selection algorithm.
It formulates layout selection as weighted partial MaxSAT and solves it with Z3.

## Requirements

- Python 3.8+
- `z3-solver` (`pip install z3-solver`)

## Usage

```bash
python3 ls_maxsat.py samples/small-single-op.txt
python3 ls_maxsat.py samples/chain-three-ops.txt --timeout 60
```

## Input Format

The solver reads the same dump format used across this project (see `samples/` for examples).
Sections: `operator_costs`, `tensor_layouts`, `tensor_transpose_costs`, `edges`.

## Encoding

Implements the formulation from `../maxsat_encoding_specification.md`:

- Boolean variables for operator configs and tensor layouts
- Hard clauses: exactly-one selection per operator/tensor, operator-tensor consistency
- Soft objective: minimize operator execution cost + transpose cost
