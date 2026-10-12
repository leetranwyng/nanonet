import copy
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.ao.quantization as quant
import torch.ao.nn.quantized as nnq

from .losses import cakd_loss


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class QuantizedWrapper(nn.Module):
    """
    Wraps a model with QuantStub / DeQuantStub so that PyTorch's static
    quantization machinery can observe or quantize the full forward pass.
    """

    def __init__(self, model):
        super().__init__()
        self.quant = quant.QuantStub()
        self.model = model
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------------------------------
# Layer manipulation helpers
# ---------------------------------------------------------------------------

def replace_relu6_with_relu(model):
    """Recursively replace all ReLU6 activations with ReLU (required for QAT)."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU6):
            setattr(model, name, nn.ReLU(inplace=module.inplace))
        else:
            replace_relu6_with_relu(module)


def patch_for_quantization(model):
    """
    Replace in-place tensor addition in residual connections with
    FloatFunctional, which is INT8-compatible.
    """
    for module in model.modules():
        if (type(module).__name__ == 'InvertedResidual'
                and getattr(module, 'use_res_connect', False)):
            module.skip_add = nnq.FloatFunctional()

            def _new_forward(self, x):
                return self.skip_add.add(x, self.conv(x))

            module.forward = types.MethodType(_new_forward, module)


# ---------------------------------------------------------------------------
# Model-specific fusion functions
# ---------------------------------------------------------------------------

def fuse_nanomobilenet(model):
    """
    Fuse Conv2d + BatchNorm2d (+ ReLU) sequences in NanoMobileNet.
    Must be called before QAT preparation.
    """
    replace_relu6_with_relu(model)
    model.eval()

    # Fuse the stem Conv-BN-ReLU
    try:
        quant.fuse_modules(model.features, ['0', '1', '2'], inplace=True)
    except Exception:
        pass

    for _name, m in model.named_modules():
        if type(m).__name__ == 'Sequential' and len(m) == 3:
            if (isinstance(m[0], nn.Conv2d)
                    and isinstance(m[1], nn.BatchNorm2d)
                    and isinstance(m[2], nn.ReLU)):
                try:
                    quant.fuse_modules(m, ['0', '1', '2'], inplace=True)
                except Exception:
                    pass

        elif type(m).__name__ == 'InvertedResidual':
            keys = list(m.conv._modules.keys())
            i = 0
            while i < len(keys):
                a = m.conv[int(keys[i])]
                if i + 2 < len(keys):
                    b, c = m.conv[int(keys[i + 1])], m.conv[int(keys[i + 2])]
                    if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and isinstance(c, nn.ReLU):
                        quant.fuse_modules(m.conv, [keys[i], keys[i + 1], keys[i + 2]], inplace=True)
                        i += 3
                        continue
                if i + 1 < len(keys):
                    b = m.conv[int(keys[i + 1])]
                    if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                        quant.fuse_modules(m.conv, [keys[i], keys[i + 1]], inplace=True)
                        i += 2
                        continue
                i += 1

    return model


def fuse_mobilenet_v2(model):
    """
    Fuse Conv2d + BatchNorm2d (+ ReLU) sequences in MobileNetV2.
    Must be called before QAT preparation.
    """
    replace_relu6_with_relu(model)
    model.eval()

    for m in model.modules():
        if type(m).__name__ == 'Conv2dNormActivation':
            to_fuse = ['0', '1'] + (['2'] if len(m) == 3 else [])
            quant.fuse_modules(m, to_fuse, inplace=True)

        elif type(m).__name__ == 'InvertedResidual':
            keys = list(m.conv._modules.keys())
            for i in range(len(keys) - 1):
                if (isinstance(m.conv._modules[keys[i]], nn.Conv2d)
                        and isinstance(m.conv._modules[keys[i + 1]], nn.BatchNorm2d)):
                    quant.fuse_modules(m.conv, [keys[i], keys[i + 1]], inplace=True)
                    break

    return model


# ---------------------------------------------------------------------------
# QAT pipeline
# ---------------------------------------------------------------------------

def run_qat(
    teacher_model,
    student_model,
    train_loader,
    val_loader,
    epochs,
    max_lr,
    patience,
    device,
    backend='qnnpack',
    fuse_fn=None,
    T=4.0,
    alpha=0.3,
):
    """
    Full Quantization-Aware Training (QAT) pipeline:

    1. Layer fusion (model-specific).
    2. Residual-add patching for INT8 compatibility.
    3. Wrap with QuantStub / DeQuantStub.
    4. Insert fake-quantize nodes via ``prepare_qat``.
    5. Calibrate observers on a few training batches.
    6. Fine-tune with CAKD.
    7. Convert to INT8 with ``quant.convert``.

    Args:
        teacher_model: Frozen teacher for CAKD distillation.
        student_model: FP32 model to quantize (copied internally; not mutated).
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        epochs:        QAT fine-tuning epochs.
        max_lr:        Peak learning rate for QAT fine-tuning.
        patience:      Early-stopping patience.
        device:        torch.device (calibration runs on device; conversion on CPU).
        backend:       Quantization backend ('qnnpack' for ARM, 'fbgemm' for x86).
        fuse_fn:       Callable ``fuse_fn(model)`` for model-specific layer fusion.
        T:             CAKD temperature.
        alpha:         CAKD KD weight.

    Returns:
        INT8 converted model (on CPU).
    """
    student_model = copy.deepcopy(student_model).to('cpu').eval()

    if fuse_fn is not None:
        fuse_fn(student_model)
    patch_for_quantization(student_model)

    wrapped = QuantizedWrapper(student_model)
    torch.backends.quantized.engine = backend
    wrapped.qconfig = quant.get_default_qat_qconfig(backend)
    qat_model = quant.prepare_qat(wrapped, inplace=False)

    # Calibrate observers with a few training batches
    qat_model.to(device).train()
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            qat_model(images.to(device))
            if i >= 10:
                break

    # CAKD fine-tuning
    teacher_model.to(device).eval()
    optimizer = optim.AdamW(qat_model.parameters(), lr=max_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        qat_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z_T = teacher_model(images)
            z_Q = qat_model(images)
            loss = cakd_loss(z_Q, z_T, labels, T=T, alpha=alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        qat_model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = qat_model(images)
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
            torch.save(qat_model.state_dict(), 'qat_best_temp.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> Early stopping at epoch {epoch+1}.")
                break

    try:
        qat_model.load_state_dict(torch.load('qat_best_temp.pth', map_location='cpu'))
    except FileNotFoundError:
        pass
    finally:
        if os.path.exists('qat_best_temp.pth'):
            os.remove('qat_best_temp.pth')

    qat_model.to('cpu').eval()
    int8_model = quant.convert(qat_model, inplace=False)
    return int8_model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model, save_path, input_shape=(1, 3, 224, 224), opset_version=13):
    """
    Export a FP32 model to ONNX format.

    Args:
        model:         FP32 PyTorch model (eval mode).
        save_path:     Output ``.onnx`` file path.
        input_shape:   Dummy input shape (N, C, H, W).
        opset_version: ONNX opset version.
    """
    model.to('cpu').eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"ONNX model exported to: {save_path} ({size_mb:.4f} MB)")
