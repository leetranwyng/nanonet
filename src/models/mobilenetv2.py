import torch.nn as nn
import torchvision.models as models


def get_mobilenet_v2_01x(num_classes=4):
    """
    Build MobileNetV2 with width_mult=0.1 (~91K parameters).

    The torchvision implementation keeps the final conv at 1280 channels
    regardless of width_mult (``last_channel = max(1280, ...)``) so the
    classifier head size is always 1280 → num_classes.

    Args:
        num_classes: Number of output classes.

    Returns:
        MobileNetV2 model instance.
    """
    model = models.mobilenet_v2(width_mult=0.1)
    model.classifier[0] = nn.Dropout(p=0.2, inplace=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model
