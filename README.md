# NanoNet: Ultra-Lightweight Brain Tumor Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

Official implementation of:

> **NanoNet: An Ultra-Lightweight Neural Architecture with Iterative Pruning and Quantization-Aware Training for Brain Tumor Classification**  
> *Submitted to SOICT 2026*

---

## Overview

NanoNet is a model compression pipeline for deploying accurate brain tumor classifiers on resource-constrained (edge) devices. Starting from a trained teacher model, the pipeline applies:

1. **Iterative structured pruning** — progressively removes less important channels using Taylor importance scores, guided by [Torch-Pruning](https://github.com/VainF/Torch-Pruning)
2. **CAKD fine-tuning** — Compressed-Aware Knowledge Distillation recovers accuracy after each pruning step
3. **Quantization-Aware Training (QAT)** — simulates INT8 inference during training to minimise accuracy drop at deployment

Two model families are provided:

| Model | Baseline Params | Baseline Acc | Pruned Params | Reduction | Pruned+QAT Acc |
|---|---|---|---|---|---|
| **NanoMobileNet** | 8,132 | 92.8% | 6,935 | −14.7% | ~91.2% |
| **MobileNetV2-0.1×** | 91,204 | 97.8% | 30,787 | −66.2% | ~93.6% |

---

## Dataset

This project uses the **BRISC-2025** brain tumor MRI dataset with four classes: `glioma`, `meningioma`, `no_tumor`, `pituitary`.

**Download:** [kaggle.com/datasets/briscdataset/brisc2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)

After downloading, extract the dataset so it matches this structure:

```
data/
└── brisc2025/
    └── classification_task/
        ├── train/
        │   ├── glioma/
        │   ├── meningioma/
        │   ├── no_tumor/
        │   └── pituitary/
        └── test/
            ├── glioma/
            ├── meningioma/
            ├── no_tumor/
            └── pituitary/
```

---

## Installation

```bash
git clone https://github.com/<your-username>/nanonet.git
cd nanonet
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA-capable GPU recommended (tested on NVIDIA Tesla T4).

---

## Running

All hyperparameters default to the exact values used in the paper. Pass `--help` to any script to see all available options.

### NanoMobileNet Pipeline

```bash
python scripts/train_nanomobilenet.py \
    --dataset-dir ./data/brisc2025/classification_task \
    --output-dir  ./outputs/nanomobilenet
```

Produces:
- `outputs/nanomobilenet/nanomobilenet_teacher.pth` — full teacher checkpoint
- `outputs/nanomobilenet/nanomobilenet_pruned.pth` — pruned FP32 checkpoint
- `outputs/nanomobilenet/nanomobilenet_int8.pth`   — INT8 quantized model

### MobileNetV2-0.1× Pipeline

```bash
python scripts/train_mobilenetv2.py \
    --dataset-dir ./data/brisc2025/classification_task \
    --output-dir  ./outputs/mobilenetv2
```

Produces:
- `outputs/mobilenetv2/mobilenetv2_teacher.pth`  — full teacher checkpoint
- `outputs/mobilenetv2/mobilenetv2_pruned.pth`   — pruned FP32 checkpoint
- `outputs/mobilenetv2/mobilenetv2_int8.pth`     — INT8 quantized model
- `outputs/mobilenetv2/mobilenetv2_pruned.onnx`  — ONNX export of pruned FP32 model

### Key Hyperparameters

| Argument | NanoMobileNet | MobileNetV2 | Description |
|---|---|---|---|
| `--teacher-epochs` | 50 | 50 | Epochs for teacher training |
| `--pruning-ratio` | 0.10 | 0.60 | Total channel sparsity target |
| `--pruning-steps` | 1 | 6 | Number of iterative pruning rounds |
| `--finetune-epochs` | 7 | 7 | CAKD recovery epochs per step |
| `--final-epochs` | 50 | 50 | Final deep CAKD fine-tuning epochs |
| `--qat-epochs` | 10 | 15 | QAT fine-tuning epochs |
| `--qat-backend` | qnnpack | qnnpack | `qnnpack` (ARM) or `fbgemm` (x86) |

---

## Project Structure

```
nanonet/
├── src/
│   ├── datasets.py          # BRISC-2025 data pipeline & augmentations
│   ├── losses.py            # CAKD loss function
│   ├── trainer.py           # Teacher training & CAKD fine-tuning loops
│   ├── pruning.py           # Iterative magnitude pruning with CAKD recovery
│   ├── quantization.py      # QAT pipeline, layer fusion, ONNX export
│   └── models/
│       ├── nanomobilenet.py # NanoMobileNet architecture (~8K params)
│       └── mobilenetv2.py   # MobileNetV2-0.1× wrapper (~91K params)
├── scripts/
│   ├── train_nanomobilenet.py   # NanoMobileNet end-to-end pipeline
│   └── train_mobilenetv2.py     # MobileNetV2 end-to-end pipeline
├── configs/
│   ├── nanomobilenet.yaml   # Reference hyperparameter values
│   └── mobilenetv2.yaml     # Reference hyperparameter values
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{nanonet2026soict,
  title     = {NanoNet: An Ultra-Lightweight Neural Architecture with Iterative Pruning
               and Quantization-Aware Training for Brain Tumor Classification},
  booktitle = {Proceedings of the 2026 Symposium on Information and Communication Technology (SOICT)},
  year      = {2026},
}
```

---

## License

This project is released under the [MIT License](LICENSE).
