# Benchmark Model Configurations

This document specifies the model architectures used in our evaluation, along with their compilation parameters. Total test variants: 217 (201 on Trainium 1, 16 on Trainium 2).

## Trainium 1

### Encoder-Only Transformers

| Model | HuggingFace ID | Batch Size | Sequence Length | TP | Auto-Cast | Opt Level |
|-------|---------------|------------|-----------------|-----|-----------|-----------|
| BERT-base | `bert-base-uncased` | 8 | 384 | 1 | matmult | — |
| BERT-large | `bert-large-uncased` | 4 | 384 | 1 | matmult | — |
| ALBERT-large | `albert-large-v2` | 4 | 512 | 1 | matmult | — |
| XLM-RoBERTa | `xlm-roberta-base` | 2 | 256 | 1 | matmult | — |
| ELECTRA-large | `google/electra-large-discriminator` | 2 | 512 | 1 | matmult | — |

### Encoder-Decoder Models

| Model | HuggingFace ID | Batch Size | Sequence Length | TP | Auto-Cast | Opt Level | Additional Flags |
|-------|---------------|------------|-----------------|-----|-----------|-----------|------------------|
| BART-base | `facebook/bart-base` | 2 | 384 | 1 | matmult | — | — |
| DistilBART | `sshleifer/distilbart-cnn-6-6` | 4 | 128 | 1 | matmult | — | — |
| Whisper-encoder | `whisper-large-v3-turbo` | 1 | 1500 | 1 | none | -O1 | `--enable-saturate-infinity`, `--enable-mixed-precision-accumulation`, `--enable-ccop-compute-overlap`, `--cc-pipeline-tiling-factor=2` |
| Whisper-decoder | `whisper-large-v3-turbo` | 1 | 100 | 1 | none | -O1 | `--enable-saturate-infinity`, `--enable-mixed-precision-accumulation`, `--enable-ccop-compute-overlap`, `--cc-pipeline-tiling-factor=2` |
| PLBART | `plbart-base` | 1 | 128 | 1 | — | — | — |
| MVP | `mvp` | 1 | 128 | 1 | — | — | — |
| MVP (ctx enc) | `mvp` | 1 | 2048 | 1 | — | — | 3 variants |
| MVP (tok gen) | `mvp` | 1 | 2048 | 1 | — | — | 2 variants |
| MusicGen (ctx enc) | `musicgen-melody` | 1 | 2048 | 1 | — | — | — |

### Decoder-Only LLMs

| Model | HuggingFace ID | Phase | Batch Size | Sequence Length | TP | Opt Level | Variants |
|-------|---------------|-------|------------|-----------------|-----|-----------|----------|
| Ministral-4b | `Ministral-4b-instruct` | ctx enc | 1 | 2048 | 1 | -O1 | 1 |
| Ministral-4b | `Ministral-4b-instruct` | tok gen | 1 | 2048 | 1 | -O2 | 1 |
| OLMo-7B | `OLMo-7B-Instruct` | ctx enc | 1 | 2048 | 1 | -O1 | 5 |
| OLMo-7B | `OLMo-7B-Instruct` | tok gen | 1 | 2048 | 1 | -O2 | 4 |
| OLMo-2-7B | `OLMo-2-1124-7B` | ctx enc | 1 | 2048 | 1 | -O1 | 1 |
| OLMo-2-7B | `OLMo-2-1124-7B` | tok gen | 1 | 2048 | 1 | -O2 | 1 |
| Granite-3.1-3b | `granite-3.1-3b-a800m-instruct` | ctx enc | 1 | 2048 | 1 | -O1 | 2 |
| Granite-3.1-3b | `granite-3.1-3b-a800m-instruct` | tok gen | 1 | 2048 | 1 | -O2 | 1 |
| VaultGemma-1b | `vaultgemma-1b` | ctx enc | 1 | 2048 | 1 | -O1 | 2 |
| VaultGemma-1b | `vaultgemma-1b` | tok gen | 1 | 2048 | 1 | -O2 | 1 |

### Mixture-of-Experts

| Model | HuggingFace ID | Phase | Batch Size | Sequence Length | TP | Opt Level | Variants |
|-------|---------------|-------|------------|-----------------|-----|-----------|----------|
| OLMoE | `OLMoE-1B-7B-0924` | ctx enc | 1 | 2048 | 1 | -O1 | 1 |
| OLMoE | `OLMoE-1B-7B-0924` | tok gen | 1 | 2048 | 1 | -O2 | 2 |
| FlexOlmo | `FlexOlmo-7x7B-1T` | ctx enc | 1 | 2048 | 1 | -O1 | 1 |
| FlexOlmo | `FlexOlmo-7x7B-1T` | tok gen | 1 | 2048 | 1 | -O2 | 1 |

### Multimodal Models

| Model | HuggingFace ID | Phase | Batch Size | Sequence Length | TP | Opt Level | Variants |
|-------|---------------|-------|------------|-----------------|-----|-----------|----------|
| Qwen2-Audio | `Qwen2-Audio-7B` | ctx enc | 1 | 2048 | 1 | -O1 | 1 |
| Qwen2-Audio | `Qwen2-Audio-7B` | tok gen | 1 | 2048 | 1 | -O2 | 1 |
| Qwen3-Next | `Qwen3-Next-80B-A3B-Instruct` | ctx enc | 1 | 2048 | 32 | -O1 | 1 |
| Qwen3-Next | `Qwen3-Next-80B-A3B-Instruct` | tok gen | 1 | 2048 | 32 | -O2 | 2 |
| Qwen3-VL | `Qwen3-VL-8B-Thinking` | ctx enc | 1 | 2048 | 1 | -O1 | 2 |
| Qwen3-VL | `Qwen3-VL-8B-Thinking` | tok gen | 1 | 2048 | 1 | -O2 | 2 |
| idefics | `idefics-9b-instruct` | tok gen | 1 | 2048 | 1 | -O2 | 2 |
| Kosmos | `kosmos-2.5` | ctx enc | 1 | 2048 | 1 | -O1 | 1 |
| PaliGemma | `paligemma-3b-mix-224` | ctx enc | 1 | 2048 | 1 | -O1 | 1 |

### CNN / Perceiver Architectures

| Model | Type | Batch Size | Framework | Auto-Cast | Notes |
|-------|------|------------|-----------|-----------|-------|
| ResNet50 | cnn-training | 16 | — | — | Standard ImageNet configuration |
| UNet | cnn-training | 4 | — | — | 3 variants |
| Vision Perceiver (conv) | — | 4 | XLA | — | Convolutional preprocessing |
| Vision Perceiver (learned) | — | — | — | all | `neuroncore_pipeline_cores=1` |
| Language Perceiver | — | 20 | — | — | Cross-attention architecture |

## Trainium 2

### Stable Diffusion Models

| Model | HuggingFace ID | Batch Size | TP | Precision | Model Type | Opt Level | Auto-Cast | Additional Flags |
|-------|---------------|------------|-----|-----------|------------|-----------|-----------|------------------|
| SD-XL UNet Refiner | `sd-xl-unet-refiner` | — | — | bfloat16 | unet-inference | — | — | — |
| SD 3.5 Large VAE | `stable-diffusion-3.5-large-vae` | 1 | 1 | bfloat16 | unet-inference | -O1 | none | — |
| SD 2 UNet 768×768 | `sd-2-unet-768x768` | — | — | bfloat16 | unet-inference | — | — | `--vectorize-strided-dma`, dual-core |
| SD 2 UNet 512×512 | `sd-2-unet-512x512` | — | — | bfloat16 | unet-inference | — | — | `--vectorize-strided-dma`, dual-core |

### Qwen 2.5 Omni 7B

| Sub-Model | Batch Size | Sequence Length | TP | Opt Level | Auto-Cast | Additional Flags |
|-----------|------------|-----------------|-----|-----------|-----------|------------------|
| Vision Encoder | — | — | — | -O1 | none | `--enable-saturate-infinity`, `--enable-mixed-precision-accumulation`, `--enable-ccop-compute-overlap`, `--cc-pipeline-tiling-factor=2` |
| Talker Model | 1 | 4096 | 1 | -O1 | none | `--enable-saturate-infinity`, `--enable-mixed-precision-accumulation`, `--enable-ccop-compute-overlap`, `--cc-pipeline-tiling-factor=2` |
| Thinker Text Model | 1 | 4096 | 1 | -O1 | none | `--enable-saturate-infinity`, `--enable-mixed-precision-accumulation`, `--enable-ccop-compute-overlap`, `--cc-pipeline-tiling-factor=2` |
| Token2Wav Model | — | — | — | -O1 | none | `--enable-saturate-infinity`, `--enable-mixed-precision-accumulation`, `--enable-ccop-compute-overlap`, `--cc-pipeline-tiling-factor=2` |

## Configuration Notes

- **Precision:** All models use BFloat16.
- **Tensor Parallelism (TP):** Most models use TP=1; Qwen3-Next uses TP=32 due to its 80B parameter size.
- **Workload Phases:** LLMs, MoE, and multimodal models are tested with separate context encoding (`-O1`) and token generation (`-O2`) phases.
- **Hardware:** Trainium 1 (trn1.32xlarge) for 201 variants; Trainium 2 (trn2.48xlarge) for 16 variants.
- **Layout Algorithms:** trn1 tests use all 3 algorithms (`greedy`, `smt`, `naive`); trn2 tests use only `greedy` and `naive`.

## Common Configuration Parameters

### Layout Algorithms Tested

- `greedy` - Greedy search with cost-based heuristics
- `smt` - SMT solver-based optimization
- `naive` - Baseline layout assignment

### Compiler Flags

- **Optimization levels:** `-O1`, `-O2`
- **Auto-cast modes:** `none`, `matmult`, `all`
- **Model types:** `transformer`, `cnn-training`, `unet-inference`

## Test Coverage

- **Total test variants:** 217 (201 on trn1, 16 on trn2)
- **Layout algorithms per model:** 3 on trn1 (greedy, smt, naive); 2 on trn2 (greedy, naive)
- **Hardware targets:** AWS Trainium (trn1, trn2)
- **Workload phases:** Context encoding and token generation for LLMs

## Notes

- All models use BFloat16 precision
- Tensor parallelism (tp) varies: tp1 for most models, tp32 for largest models
- Some models tested with multiple module variants to cover different graph structures
- Test infrastructure uses pytest with parameterization for systematic coverage
