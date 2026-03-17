# Overview

This artifact provides specifications, benchmark configurations, and algorithmic descriptions for reproducing the layout selection techniques presented in our paper. Due to proprietary constraints, we provide abstract specifications rather than the full AWS Neuron SDK implementation.

## Artifact Contents

```
artifact/
├── README.md
├── benchmarks/
│   ├── model_configurations.md         # Detailed benchmark specifications
│   └── test_parameters.json            # Batch sizes, sequence lengths, etc.
├── algorithms/
│   ├── treewidth_dp_specification.md
│   ├── maxsat_encoding_specification.md
│   └── greedy_randomized_search.md
├── verification/
│   └── layout_verification_spec.md     # Verification pass specification
├── cost_model/
│   └── cost_model_interface.md         # Abstract cost model API
├── experiments/
│   └── evaluation_metrics.md           # Metrics and analysis approach
└── LICENSE
```

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

## Algorithms

### Treewidth-Based Dynamic Programming

The `algorithms/treewidth_dp_specification.md` file provides:

- Formal algorithm specification
- DP table construction procedure
- Inductive computation steps
- Runtime complexity analysis

### MaxSAT Encoding

The `algorithms/maxsat_encoding_specification.md` file provides:

- Operator variable encoding
- Transpose variable encoding
- Constraint formulation
- Integration with Z3 solver

### Greedy Search with Randomization

The `algorithms/greedy_randomized_search.md` file provides:

- Sequential exhaustive search algorithm
- Near-optimal solution enumeration
- Randomness tolerance parameter (0.0-1.0)
- Solution selection strategy

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

## Experimental Methodology

The `experimental_setup/` directory contains:

- Hardware platform specifications (AWS Trainium)
- Compilation workflow and flags
- Layout algorithm configurations ("greedy", "smt", "naive")
- Performance metrics and analysis procedures

## Reproducing Results

### Prerequisites

- Access to AWS Trainium instances (trn1.32xlarge or trn2.48xlarge)
- AWS Neuron SDK
- Python 3.8+, PyTorch 2.0+

### Steps

1. Download benchmark models from HuggingFace using configurations in `benchmarks/`
2. Implement the cost model interface for your target hardware
3. Implement the algorithms following specifications in `algorithms/`
4. Run compilation with layout selection enabled
5. Measure execution time and compare against baselines

> **Note:** Full reproduction requires access to AWS Trainium hardware and the Neuron SDK. The algorithmic contributions can be applied to other ML compiler frameworks by implementing the cost model interface.

## Limitations

- **Proprietary Components:** AWS Trainium's internal cost model, hardware latency tables, and precomputed cost data are not included
- **Hardware Access:** Results are specific to AWS Trainium; performance on other hardware may vary
- **Closed-Source SDK:** The AWS Neuron SDK is publicly available but not open-source

## Citation

If you use this artifact, please cite our paper:

```bibtex
@inproceedings{layout_selection_oopsla2026,
    title = {Tensor Seeks Layout: Finding the Right Memory Layout for ML Compilers},
    author = {[Authors]},
    booktitle = {Proceedings of the ACM on Programming Languages (OOPSLA)},
    year = {2026}
}
```

## Contact

For questions about this artifact or the paper, please contact: [author emails]

## License

This artifact is released under the MIT License. See [LICENSE](LICENSE) file for details.
