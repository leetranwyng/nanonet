import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .losses import cakd_loss


def evaluate(model, loader, device):
    """
    Evaluate model accuracy and macro F1-score on a DataLoader.

    Returns:
        Tuple of (accuracy, f1_score).
    """
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, preds = torch.max(model(images), dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    return acc, f1


def train_teacher(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    checkpoint_path='teacher_best.pth',
    weight_decay=1e-4,
):
    """
    Train a model from scratch using cross-entropy loss with cosine LR annealing.
    Saves the best checkpoint (by validation accuracy) and restores it on exit.

    Args:
        model:           Model to train (already on device).
        train_loader:    Training DataLoader.
        val_loader:      Validation DataLoader.
        epochs:          Number of training epochs.
        lr:              Peak learning rate.
        device:          torch.device.
        checkpoint_path: Path to save the best model state dict.
        weight_decay:    AdamW weight decay.

    Returns:
        Model loaded with the best validation checkpoint.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        acc, f1 = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.2e} "
            f"| Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def finetune_with_cakd(
    teacher_model,
    student_model,
    train_loader,
    val_loader,
    epochs,
    max_lr,
    patience=3,
    device='cpu',
    checkpoint_path='temp_best.pth',
    T=4.0,
    alpha=0.3,
):
    """
    Fine-tune a student model using CAKD against a frozen teacher.

    Uses AdamW + ReduceLROnPlateau + early stopping (patience on val loss).
    Restores the best checkpoint before returning.

    Args:
        teacher_model:    Frozen teacher model.
        student_model:    Student/pruned model to fine-tune.
        train_loader:     Training DataLoader.
        val_loader:       Validation DataLoader.
        epochs:           Maximum number of fine-tuning epochs.
        max_lr:           Initial learning rate.
        patience:         Early-stopping patience (number of epochs without
                          val loss improvement before stopping).
        device:           torch.device.
        checkpoint_path:  Temporary path to save the best student weights.
        T:                CAKD distillation temperature.
        alpha:            CAKD KD loss weight.

    Returns:
        Student model restored to its best validation checkpoint.
    """
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    optimizer = optim.AdamW(student_model.parameters(), lr=max_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z_T = teacher_model(images)
            z_S = student_model(images)
            loss = cakd_loss(z_S, z_T, labels, T=T, alpha=alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        student_model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                val_loss += F.cross_entropy(outputs, labels).item() * images.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"  Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.2e} "
            f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(student_model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> Early stopping at epoch {epoch+1}.")
                break

    try:
        student_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        pass

    return student_model
