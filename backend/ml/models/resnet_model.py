from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torchvision import models


class PlantDiseaseResNet(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_layers: int = 6,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base    = models.resnet18(weights=weights)

        if freeze_layers > 0:
            children = list(base.children())
            for layer in children[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Store layers individually instead of nn.Sequential
        # This avoids the view/inplace issue with Sequential wrapping
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(base.fc.in_features, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.classifier(x)

    def get_last_conv_layer(self) -> nn.Module:
        """
        Return conv2 of the last BasicBlock in layer4.
        Stored directly as self.layer4 now — no Sequential indexing needed.
        """
        last_block = self.layer4[-1]
        return last_block.conv2

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def save_checkpoint(
    model: PlantDiseaseResNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    class_mapping=None,
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":         metrics,
        "num_classes":     model.num_classes,
        "class_mapping":   class_mapping,
    }, save_path)
    print(f"[Checkpoint] Saved → {save_path}")


def _remap_state_dict(state_dict: dict) -> dict:
    """
    Remap keys from old checkpoint formats to the current model layout.

    Handles two legacy formats:
      - 'features.N.*'  → layer/conv keys used by the current PlantDiseaseResNet
      - 'model.*'       → top-level 'model.' prefix sometimes added by trainers
    """
    # Map from old 'features.N' index → current attribute name
    FEATURES_MAP = {
        "0": "conv1",
        "1": "bn1",
        "2": "relu",
        "3": "maxpool",
        "4": "layer1",
        "5": "layer2",
        "6": "layer3",
        "7": "layer4",
        "8": "avgpool",
    }

    needs_remap = any(
        k.startswith("features.") or k.startswith("model.")
        for k in state_dict
    )
    if not needs_remap:
        return state_dict  # Already in the correct format

    remapped = {}
    for old_key, value in state_dict.items():
        new_key = old_key

        # Strip leading 'model.' prefix if present
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        # Remap 'features.N.rest' → '<layer_name>.rest'
        if new_key.startswith("features."):
            parts = new_key.split(".", 2)   # ["features", "N", "rest..."]
            idx   = parts[1]
            rest  = parts[2] if len(parts) > 2 else ""
            if idx in FEATURES_MAP:
                mapped  = FEATURES_MAP[idx]
                new_key = f"{mapped}.{rest}" if rest else mapped

        remapped[new_key] = value

    return remapped


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_classes: Optional[int] = None,
) -> tuple:
    ckpt  = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    n_cls = num_classes or ckpt["num_classes"]
    model = PlantDiseaseResNet(num_classes=n_cls, pretrained=False)

    raw_state = ckpt["model_state"]
    remapped  = _remap_state_dict(raw_state)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"[Checkpoint] WARNING – missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[Checkpoint] WARNING – unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model, ckpt


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple Silicon MPS")
    else:
        d = torch.device("cpu")
        print("[Device] CPU")
    return d