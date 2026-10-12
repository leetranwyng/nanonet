import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

from .trainer import finetune_with_cakd


def run_iterative_pruning(
    teacher_model,
    student_model,
    train_loader,
    val_loader,
    pruning_ratio=0.6,
    iterative_steps=6,
    finetune_epochs=7,
    finetune_lr=5e-4,
    finetune_patience=3,
    final_finetune_epochs=50,
    final_finetune_lr=1e-4,
    final_finetune_patience=10,
    device='cpu',
):
    """
    Iterative magnitude pruning with CAKD recovery after each step.

    Uses TaylorImportance for filter ranking and MagnitudePruner for
    structured channel removal. After all steps, a final deep fine-tune
    is performed.

    Args:
        teacher_model:          Frozen teacher (used for CAKD distillation).
        student_model:          Model to prune (modified in place).
        train_loader:           Training DataLoader.
        val_loader:             Validation DataLoader.
        pruning_ratio:          Total channel sparsity target (0–1).
        iterative_steps:        Number of pruning rounds to divide target into.
        finetune_epochs:        Epochs of CAKD recovery after each prune step.
        finetune_lr:            Learning rate for per-step CAKD recovery.
        finetune_patience:      Early-stopping patience per step.
        final_finetune_epochs:  Epochs of deep CAKD fine-tuning after all steps.
        final_finetune_lr:      Learning rate for the final fine-tune.
        final_finetune_patience: Early-stopping patience for the final fine-tune.
        device:                 torch.device.

    Returns:
        Pruned and fine-tuned student model.
    """
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    ignored_layers = [m for m in student_model.modules() if isinstance(m, nn.Linear)]

    base_macs, base_params = tp.utils.count_ops_and_params(student_model, example_inputs)
    print(f"[Before Pruning] Params: {base_params:,} | MACs: {base_macs:,}")

    imp = tp.importance.TaylorImportance()
    pruner = tp.pruner.MagnitudePruner(
        model=student_model,
        example_inputs=example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=pruning_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
    )

    for step in range(iterative_steps):
        print(f"\n--- Pruning Step {step + 1}/{iterative_steps} ---")

        # Compute Taylor importance scores via a single forward-backward pass
        student_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(student_model(images), labels)
            loss.backward()
            break

        pruner.step()
        student_model.zero_grad()

        macs, nparams = tp.utils.count_ops_and_params(student_model, example_inputs)
        print(f"-> Current size: Params: {nparams:,} | MACs: {macs:,}")

        print("-> CAKD recovery fine-tuning...")
        finetune_with_cakd(
            teacher_model, student_model,
            train_loader, val_loader,
            epochs=finetune_epochs,
            max_lr=finetune_lr,
            patience=finetune_patience,
            device=device,
        )
        student_model.zero_grad()

    print("\n-> Final deep CAKD fine-tuning...")
    finetune_with_cakd(
        teacher_model, student_model,
        train_loader, val_loader,
        epochs=final_finetune_epochs,
        max_lr=final_finetune_lr,
        patience=final_finetune_patience,
        device=device,
    )

    return student_model
