from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

from .io_utils import read_csv


LABEL_NAME_KEYS = {
    "team": "team_names",
    "driver": "driver_names",
    "car_model": "car_model_names",
}


def build_backbone(backbone_name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    raise ValueError(f"Unsupported backbone: {backbone_name}")


class MultiHeadClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_team_classes: int,
        num_driver_classes: int,
        num_car_model_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.dropout = nn.Dropout(p=dropout)
        self.team_head = nn.Linear(feature_dim, num_team_classes)
        self.driver_head = nn.Linear(feature_dim, num_driver_classes)
        self.car_model_head = nn.Linear(feature_dim, num_car_model_classes)

    def forward(self, images: torch.Tensor, heads: list[str] | tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        features = self.dropout(features)
        outputs = {}
        selected_heads = heads or ("team", "driver", "car_model")
        head_modules = {
            "team": self.team_head,
            "driver": self.driver_head,
            "car_model": self.car_model_head,
        }
        for head in selected_heads:
            outputs[head] = head_modules[head](features)
        return outputs


class CropClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform: transforms.Compose,
        cache_mode: str = "none",
    ) -> None:
        self.rows = [row for row in read_csv(manifest_path) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows with split={split} found in {manifest_path}")
        self.transform = transform
        self.cache_mode = cache_mode
        if cache_mode not in {"none", "ram"}:
            raise ValueError(f"Unsupported cache_mode={cache_mode!r}")
        self.cached_images: list[Image.Image] | None = None
        if cache_mode == "ram":
            self.cached_images = []
            for row in self.rows:
                with Image.open(row["crop_path"]) as image_file:
                    self.cached_images.append(image_file.convert("RGB"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if self.cached_images is not None:
            image = self.cached_images[index].copy()
        else:
            with Image.open(row["crop_path"]) as image_file:
                image = image_file.convert("RGB")
        image_tensor = self.transform(image)
        return {
            "image": image_tensor,
            "team_target": int(row["team_id"]),
            "driver_target": int(row["driver_id"]),
            "car_model_target": int(row["car_model_id"]),
            "row": row,
        }


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform


def build_eval_transform(image_size: int) -> transforms.Compose:
    return build_transforms(image_size=image_size)[1]


@dataclass
class LoadedClassifier:
    model: MultiHeadClassifier
    label_maps: dict
    image_size: int
    heads: list[str]


def load_classifier_checkpoint(checkpoint_path: Path, device: torch.device) -> LoadedClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_maps = checkpoint["label_maps"]
    image_size = int(checkpoint.get("image_size", 384))
    heads = list(checkpoint.get("heads", ["team", "driver", "car_model"]))
    model = MultiHeadClassifier(
        backbone_name=checkpoint["backbone"],
        num_team_classes=len(label_maps["team_names"]),
        num_driver_classes=len(label_maps["driver_names"]),
        num_car_model_classes=len(label_maps["car_model_names"]),
        pretrained=False,
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)
    return LoadedClassifier(model=model, label_maps=label_maps, image_size=image_size, heads=heads)


def classify_crops(
    crops: list[Image.Image],
    model: MultiHeadClassifier,
    label_maps: dict,
    heads: list[str],
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 32,
) -> list[dict]:
    if not crops:
        return []

    results: list[dict] = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch_crops = crops[start : start + batch_size]
            images = torch.stack([transform(crop) for crop in batch_crops]).to(device)
            outputs = model(images, heads=heads)
            batch_results = [{} for _ in batch_crops]
            for head in heads:
                probabilities = torch.softmax(outputs[head], dim=1)
                confidences, class_ids = probabilities.max(dim=1)
                names = label_maps[LABEL_NAME_KEYS[head]]
                for index, (confidence, class_id) in enumerate(zip(confidences, class_ids, strict=True)):
                    batch_results[index][head] = names[int(class_id.item())]
                    batch_results[index][f"{head}_confidence"] = round(float(confidence.item()), 5)
            results.extend(batch_results)
    return results
