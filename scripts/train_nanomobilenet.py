#!/usr/bin/env python3
"""
NanoMobileNet training pipeline:
  Phase 1 — Train teacher from scratch
  Phase 2 — Iterative pruning + CAKD recovery
  Phase 3 — Quantization-Aware Training (INT8, qnnpack)

Usage:
    python scripts/train_nanomobilenet.py --dataset-dir <path/to/classification_task>
"""

import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import build_dataloaders
from src.models.nanomobilenet import NanoMobileNet, kaiming_init
from src.trainer import train_teacher, evaluate
from src.pruning import run_iterative_pruning
from src.quantization import run_qat, fuse_nanomobilenet


def parse_args():
    p = argparse.ArgumentParser(description="NanoMobileNet Training Pipeline")
    p.add_argument('--dataset-dir', type=str, required=True,
                   help='Path to the BRISC-2025 classification_task directory')
    p.add_argument('--output-dir', type=str, default='./outputs/nanomobilenet')

    # Data
    p.add_argument('--num-classes', type=int, default=4)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=0)

    # Teacher
    p.add_argument('--teacher-epochs', type=int, default=50)
    p.add_argument('--teacher-lr', type=float, default=1e-3)

    # Pruning
    p.add_argument('--pruning-ratio', type=float, default=0.1,
                   help='Channel sparsity (0–1). Paper uses 0.10 for NanoMobileNet.')
    p.add_argument('--pruning-steps', type=int, default=1)
    p.add_argument('--finetune-epochs', type=int, default=7)
    p.add_argument('--finetune-lr', type=float, default=5e-4)
    p.add_argument('--finetune-patience', type=int, default=2)
    p.add_argument('--final-epochs', type=int, default=50)
    p.add_argument('--final-lr', type=float, default=1e-4)
    p.add_argument('--final-patience', type=int, default=5)

    # QAT
    p.add_argument('--qat-epochs', type=int, default=10)
    p.add_argument('--qat-lr', type=float, default=1e-5)
    p.add_argument('--qat-patience', type=int, default=3)
    p.add_argument('--qat-backend', type=str, default='qnnpack',
                   choices=['qnnpack', 'fbgemm'],
                   help='qnnpack = ARM devices, fbgemm = x86')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        args.dataset_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # -------------------------------------------------------------------------
    # Phase 1: Train teacher from scratch
    # -------------------------------------------------------------------------
    print("\n========== Phase 1: Teacher Training ==========")
    teacher = NanoMobileNet(num_classes=args.num_classes).to(device)
    teacher.apply(kaiming_init)

    teacher_ckpt = os.path.join(args.output_dir, 'nanomobilenet_teacher.pth')
    teacher = train_teacher(
        teacher, train_loader, val_loader,
        epochs=args.teacher_epochs,
        lr=args.teacher_lr,
        device=device,
        checkpoint_path=teacher_ckpt,
    )

    acc, f1 = evaluate(teacher, test_loader, device)
    print(f"[Teacher] Test Accuracy: {acc:.4f} | Test F1: {f1:.4f}")

    # -------------------------------------------------------------------------
    # Phase 2: Iterative pruning + CAKD
    # -------------------------------------------------------------------------
    print("\n========== Phase 2: Iterative Pruning + CAKD ==========")
    student = copy.deepcopy(teacher)
    student = run_iterative_pruning(
        teacher_model=teacher,
        student_model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        pruning_ratio=args.pruning_ratio,
        iterative_steps=args.pruning_steps,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_patience=args.finetune_patience,
        final_finetune_epochs=args.final_epochs,
        final_finetune_lr=args.final_lr,
        final_finetune_patience=args.final_patience,
        device=device,
    )

    pruned_ckpt = os.path.join(args.output_dir, 'nanomobilenet_pruned.pth')
    torch.save(student.state_dict(), pruned_ckpt)

    acc, f1 = evaluate(student, test_loader, device)
    print(f"[Pruned FP32] Test Accuracy: {acc:.4f} | Test F1: {f1:.4f}")

    # -------------------------------------------------------------------------
    # Phase 3: Quantization-Aware Training
    # -------------------------------------------------------------------------
    print("\n========== Phase 3: Quantization-Aware Training (QAT) ==========")
    int8_model = run_qat(
        teacher_model=teacher,
        student_model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.qat_epochs,
        max_lr=args.qat_lr,
        patience=args.qat_patience,
        device=device,
        backend=args.qat_backend,
        fuse_fn=fuse_nanomobilenet,
    )

    int8_path = os.path.join(args.output_dir, 'nanomobilenet_int8.pth')
    torch.save(int8_model.state_dict(), int8_path)

    size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"\nINT8 model saved to: {int8_path} ({size_mb:.4f} MB)")


if __name__ == '__main__':
    main()
