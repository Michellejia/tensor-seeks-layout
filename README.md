# Tensor Seeks Layout: Finding the Right Memory Layout for ML Compilers

This artifact provides algorithm implementations, specifications, benchmark configurations, and experimental data for reproducing the layout selection techniques presented in our paper.

## Artifact Contents

```
├── algorithms/
│   ├── treewidth_dp_specification.md       # Formal spec for treewidth-based DP
│   ├── maxsat_encoding_specification.md    # Formal spec for MaxSAT encoding
│   ├── greedy_randomized_search.md         # Formal spec for greedy search
│   ├── treewidth_solver/                   # C++ exact solver (treewidth DP)
│   │   ├── src/ls_opt.cpp
│   │   ├── samples/
│   │   ├── Makefile
│   │   └── run_layout_selection.sh
│   ├── maxsat_solver/                      # Python exact solver (Z3 MaxSAT)
│   │   ├── ls_maxsat.py
│   │   └── samples/
│   └── greedy/                             # Python greedy search
│       ├── LayoutSelection.py
│       └── generate_test_inputs.py
├── benchmarks/
│   ├── model_configurations.md             # Model specs and compiler flags
│   └── test_parameters.json                # Batch sizes, sequence lengths, etc.
├── cost_model/
│   └── cost_model_interface.md             # Abstract cost model API
├── verification/
│   └── layout_verification_spec.md         # Verification pass specification
├── experiments/
│   ├── evaluation_metrics.md               # Metrics and analysis approach
│   ├── parse_data.py                       # Log parsing and CSV extraction
│   ├── e2e_metrics_greedy.csv              # Greedy algorithm results
│   ├── e2e_metrics_baseline.csv            # Baseline (naive) results
│   ├── greedy.log                          # Raw compilation logs (greedy)
│   └── baseline.log                        # Raw compilation logs (baseline)
└── LICENSE
```

## Algorithms

### Treewidth-Based Dynamic Programming

Exact solver using DP over a nice tree decomposition of the interaction graph. See `algorithms/treewidth_dp_specification.md` for the formal spec and `algorithms/treewidth_solver/` for the C++ implementation.

```bash
cd algorithms/treewidth_solver
make
./run_layout_selection.sh samples/ls-0660a9-smt-sg0001.txt
```

Requires: Linux, g++ (C++17), Java (for external treewidth solver).

### MaxSAT Encoding

Exact solver that encodes layout selection as weighted partial MaxSAT and solves with Z3. See `algorithms/maxsat_encoding_specification.md` for the formal spec and `algorithms/maxsat_solver/` for the implementation.

```bash
cd algorithms/maxsat_solver
python3 ls_maxsat.py samples/chain-three-ops.txt
```

Requires: Python 3.8+, `z3-solver`.

### Greedy Search with Randomization

Exhaustive search over partition axes groups with early pruning and optional randomization. See `algorithms/greedy_randomized_search.md` for the formal spec and `algorithms/greedy/` for the implementation.

```bash
cd algorithms/greedy
python3 LayoutSelection.py
```

Requires: Python 3.8+.

## Benchmark Models

Our evaluation includes 32 distinct model architectures spanning:

- **Encoder-only Transformers:** BERT-base, BERT-large, ALBERT-large-v2, XLM-RoBERTa-base, ELECTRA-large
- **Encoder-Decoder Models:** BART-base, DistilBART, Whisper-large-v3-turbo, PLBART, MVP, MusicGen
- **Decoder-only LLMs:** Ministral-4b, OLMo-7B, OLMo-2-7B, Granite-3.1-3b, VaultGemma-1b
- **Mixture-of-Experts:** OLMoE-1B-7B, FlexOlmo-7x7B
- **Multimodal Models:** Qwen2-Audio-7B, Qwen3-Next-80B, Qwen3-VL-8B, idefics-9b, Kosmos-2.5, PaliGemma-3b
- **CNN Architectures:** ResNet50, UNet, Vision Perceiver

All models are publicly available from HuggingFace: https://huggingface.co/models

See `benchmarks/model_configurations.md` for complete specifications including batch sizes, sequence lengths, and tensor parallelism settings.

## Cost Model Interface

The `cost_model/cost_model_interface.md` file specifies an abstract API that ML compiler frameworks can implement:

```python
class AbstractCostModel:
    def operator_cost(self, op, layout) -> float
    def transpose_cost(self, tensor, src_layout, dst_layout) -> float
    def is_feasible(self, op, layout) -> bool
```

> **Note:** The actual AWS Trainium cost model implementation is proprietary and not included in this artifact. However, the mathematical formulation in Section 3 of the paper provides sufficient detail for implementing cost models for other hardware targets.

## Verification Pass

The `verification/layout_verification_spec.md` file describes:

- Partition axes validation logic
- Tensor contract operation checks
- Tripcount constraint verification
- Integration into compiler pipelines

## Experimental Data

The `experiments/` directory contains:

- `evaluation_metrics.md` — hardware platform specs, compilation workflow, and analysis procedures
- `e2e_metrics_greedy.csv` / `e2e_metrics_baseline.csv` — end-to-end compilation metrics
- `greedy.log` / `baseline.log` — raw compilation logs
- `parse_data.py` — script to extract metrics from logs into CSV

Layout algorithm configurations: `"greedy"`, `"smt"`, `"naive"`.

## Reproducing Results

### Prerequisites

- Access to AWS Trainium instances (trn1.32xlarge or trn2.48xlarge)
- [AWS Neuron SDK](https://aws.amazon.com/ai/machine-learning/neuron/) (tested with v2.26.0, latest: v2.28.1)
- Python 3.8+, PyTorch 2.0+

### Steps

1. Download benchmark models from HuggingFace using configurations in `benchmarks/`
2. Implement the cost model interface for your target hardware
3. Run the solvers in `algorithms/` on your compilation instances
4. Measure execution time and compare against baselines

> **Note:** Full reproduction requires access to AWS Trainium hardware and the Neuron SDK. The algorithmic contributions can be applied to other ML compiler frameworks by implementing the cost model interface.

## Limitations

- **Proprietary Components:** AWS Trainium's internal cost model, hardware latency tables, and precomputed cost data are not included
- **Hardware Access:** Results are specific to AWS Trainium; performance on other hardware may vary
- **Closed-Source SDK:** The AWS Neuron SDK is publicly available but not open-source

## License

This artifact is released under the MIT License. See [LICENSE](LICENSE) file for details.
