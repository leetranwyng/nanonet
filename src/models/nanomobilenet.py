import torch.nn as nn


class InvertedResidual(nn.Module):
    """
    MobileNetV2-style inverted residual block with depthwise separable convolutions.
    Uses a residual shortcut when input and output dimensions match.
    """

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class NanoMobileNet(nn.Module):
    """
    NanoMobileNet: ultra-lightweight MobileNet variant (~8K parameters).

    Architecture uses progressively wider inverted residual blocks with
    aggressive downsampling at every stage. Designed for edge deployment.

    Default configuration (num_classes=4, 224×224 input):
        Stem         : 3 → 8 channels, stride 2
        Block 1      : t=1, c=8,  n=1, s=2
        Block 2      : t=2, c=12, n=1, s=2
        Block 3      : t=2, c=16, n=1, s=2
        Block 4      : t=2, c=24, n=2, s=2
        Head Conv    : 24 → 48 channels
        Classifier   : GlobalAvgPool → Dropout(0.1) → Linear
    Total: ~8,132 parameters
    """

    _SETTINGS = [
        # (expand_ratio, out_channels, num_blocks, stride)
        (1,  8, 1, 2),
        (2, 12, 1, 2),
        (2, 16, 1, 2),
        (2, 24, 2, 2),
    ]
    _INPUT_CHANNEL = 8
    _LAST_CHANNEL = 48

    def __init__(self, num_classes=4):
        super().__init__()
        inp = self._INPUT_CHANNEL

        features = [
            nn.Conv2d(3, inp, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
        ]
        for t, c, n, s in self._SETTINGS:
            for i in range(n):
                features.append(InvertedResidual(inp, c, stride=(s if i == 0 else 1), expand_ratio=t))
                inp = c

        features.append(nn.Sequential(
            nn.Conv2d(inp, self._LAST_CHANNEL, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self._LAST_CHANNEL),
            nn.ReLU6(inplace=True),
        ))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(self._LAST_CHANNEL, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def kaiming_init(model):
    """Apply Kaiming normal initialization to Conv2d and Linear layers."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
