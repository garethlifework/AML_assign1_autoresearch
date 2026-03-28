from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .config import ExperimentConfig
from .utils import ensure_dir

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(slots=True)
class OxfordPetSample:
    image_id: str
    image_path: Path
    breed_idx: int
    breed_name: str
    species_idx: int
    bbox_path: Path
    trimap_path: Path


@dataclass(slots=True)
class DatasetMetadata:
    class_names: list[str]
    species_names: list[str]
    global_class_to_species: dict[int, int]


class OxfordPetDataset(Dataset):
    def __init__(
        self,
        samples: list[OxfordPetSample],
        metadata: DatasetMetadata,
        transform: Callable | None,
        image_mode: str,
        task: str,
        return_metadata: bool = False,
    ) -> None:
        self.samples = samples
        self.metadata = metadata
        self.transform = transform
        self.image_mode = image_mode
        self.task = task
        self.return_metadata = return_metadata
        self.global_to_local = {
            breed_idx: idx
            for idx, breed_idx in enumerate(sorted({sample.breed_idx for sample in samples}))
        }
        self.local_to_global = {v: k for k, v in self.global_to_local.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw(self, sample: OxfordPetSample) -> Image.Image:
        return Image.open(sample.image_path).convert("RGB")

    def _crop_bbox(self, image: Image.Image, sample: OxfordPetSample, pad_ratio: float = 0.08) -> Image.Image:
        if not sample.bbox_path.exists():
            return image
        root = ET.parse(sample.bbox_path).getroot()
        bbox = root.find(".//bndbox")
        if bbox is None:
            return image
        xmin = int(bbox.findtext("xmin", "0"))
        ymin = int(bbox.findtext("ymin", "0"))
        xmax = int(bbox.findtext("xmax", str(image.width)))
        ymax = int(bbox.findtext("ymax", str(image.height)))
        width = max(1, xmax - xmin)
        height = max(1, ymax - ymin)
        pad_x = int(width * pad_ratio)
        pad_y = int(height * pad_ratio)
        crop = (
            max(0, xmin - pad_x),
            max(0, ymin - pad_y),
            min(image.width, xmax + pad_x),
            min(image.height, ymax + pad_y),
        )
        return image.crop(crop)

    def _trimap_foreground(self, image: Image.Image, sample: OxfordPetSample, pad_ratio: float = 0.08) -> Image.Image:
        if not sample.trimap_path.exists():
            return image
        trimap = Image.open(sample.trimap_path)
        mask = np.array(trimap)
        foreground = mask != 2
        coords = np.argwhere(foreground)
        if coords.size == 0:
            return image
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        pad_x = int(width * pad_ratio)
        pad_y = int(height * pad_ratio)
        crop = (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(image.width, x1 + pad_x),
            min(image.height, y1 + pad_y),
        )
        image = image.crop(crop)
        trimap = trimap.crop(crop)
        trimap_arr = np.array(trimap)
        img_arr = np.array(image).astype(np.float32)
        bg_mask = trimap_arr == 2
        img_arr[bg_mask] = 0.5 * img_arr[bg_mask] + 0.5 * 127.0
        return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

    def _load_image(self, sample: OxfordPetSample) -> Image.Image:
        image = self._load_raw(sample)
        if self.image_mode == "bbox":
            return self._crop_bbox(image, sample)
        if self.image_mode == "trimap_foreground":
            return self._trimap_foreground(image, sample)
        return image

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample)
        if self.transform is not None:
            image = self.transform(image)

        if self.task == "species":
            label = sample.species_idx
        else:
            label = self.global_to_local[sample.breed_idx]

        if not self.return_metadata:
            return image, label

        metadata = {
            "image_id": sample.image_id,
            "breed_global": sample.breed_idx,
            "breed_name": sample.breed_name,
            "species_idx": sample.species_idx,
        }
        return image, label, metadata


def _titleize_breed(image_id: str) -> str:
    breed_name = image_id.rsplit("_", 1)[0].replace("_", " ")
    return breed_name.title()


def load_metadata(data_dir: Path) -> DatasetMetadata:
    list_path = data_dir / "annotations" / "list.txt"
    rows = [
        line.strip().split()
        for line in list_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    class_by_idx: dict[int, str] = {}
    species_raw_by_class: dict[int, int] = {}
    for image_id, class_id, species_id, *_ in rows:
        class_idx = int(class_id) - 1
        class_by_idx[class_idx] = _titleize_breed(image_id)
        species_raw_by_class[class_idx] = int(species_id)
    species_values = sorted(set(species_raw_by_class.values()))
    species_norm = {raw: idx for idx, raw in enumerate(species_values)}
    class_names = [class_by_idx[idx] for idx in sorted(class_by_idx)]
    global_class_to_species = {
        class_idx: species_norm[species_raw]
        for class_idx, species_raw in species_raw_by_class.items()
    }
    species_names = ["cat", "dog"] if len(species_values) == 2 else [str(v) for v in species_values]
    return DatasetMetadata(
        class_names=class_names,
        species_names=species_names,
        global_class_to_species=global_class_to_species,
    )


def load_split_samples(
    data_dir: Path,
    split: str,
    metadata: DatasetMetadata,
    species_filter: int | None = None,
) -> list[OxfordPetSample]:
    split_path = data_dir / "annotations" / f"{split}.txt"
    rows = [line.strip().split() for line in split_path.read_text().splitlines() if line.strip()]
    raw_species_values = sorted({int(row[2]) for row in rows})
    species_norm = {raw: idx for idx, raw in enumerate(raw_species_values)}
    samples: list[OxfordPetSample] = []
    for image_id, class_id, species_id, *_ in rows:
        class_idx = int(class_id) - 1
        species_idx = species_norm[int(species_id)]
        if species_filter is not None and species_idx != species_filter:
            continue
        samples.append(
            OxfordPetSample(
                image_id=image_id,
                image_path=data_dir / "images" / f"{image_id}.jpg",
                breed_idx=class_idx,
                breed_name=metadata.class_names[class_idx],
                species_idx=species_idx,
                bbox_path=data_dir / "annotations" / "xmls" / f"{image_id}.xml",
                trimap_path=data_dir / "annotations" / "trimaps" / f"{image_id}.png",
            )
        )
    return samples


def stratified_train_val_indices(samples: list[OxfordPetSample], val_fraction: float, split_seed: int) -> tuple[list[int], list[int]]:
    labels = np.array([sample.breed_idx for sample in samples])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=split_seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(samples)), labels))
    return train_idx.tolist(), val_idx.tolist()


def build_transforms(config: ExperimentConfig, training: bool) -> transforms.Compose:
    if not training:
        return transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    ops: list[Callable] = []
    if config.augmentation in {"mild", "mild_erasing", "mixup", "cutmix"}:
        ops.extend(
            [
                transforms.RandomResizedCrop(config.image_size, scale=(0.88, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
            ]
        )
    else:
        ops.append(transforms.Resize((config.image_size, config.image_size)))

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    if config.augmentation == "mild_erasing" or config.random_erasing_p > 0:
        ops.append(transforms.RandomErasing(p=max(config.random_erasing_p, 0.10), scale=(0.02, 0.10)))
    return transforms.Compose(ops)


def build_datasets(config: ExperimentConfig) -> dict[str, object]:
    data_dir = config.paths.data_path()
    metadata = load_metadata(data_dir)
    trainval_samples = load_split_samples(
        data_dir=data_dir,
        split="trainval",
        metadata=metadata,
        species_filter=config.species_filter,
    )
    test_samples = load_split_samples(
        data_dir=data_dir,
        split="test",
        metadata=metadata,
        species_filter=config.species_filter,
    )
    train_idx, val_idx = stratified_train_val_indices(trainval_samples, config.val_fraction, config.split_seed)

    base_train_dataset = OxfordPetDataset(
        samples=trainval_samples,
        metadata=metadata,
        transform=build_transforms(config, training=True),
        image_mode=config.image_mode,
        task=config.task,
    )
    val_dataset_full = OxfordPetDataset(
        samples=trainval_samples,
        metadata=metadata,
        transform=build_transforms(config, training=False),
        image_mode=config.image_mode,
        task=config.task,
        return_metadata=True,
    )
    test_dataset = OxfordPetDataset(
        samples=test_samples,
        metadata=metadata,
        transform=build_transforms(config, training=False),
        image_mode=config.image_mode,
        task=config.task,
        return_metadata=True,
    )
    train_dataset = Subset(base_train_dataset, train_idx)
    val_dataset = Subset(val_dataset_full, val_idx)

    classes_present = sorted({sample.breed_idx for sample in trainval_samples}) if config.task == "breed" else sorted({sample.species_idx for sample in trainval_samples})
    label_names = (
        [metadata.class_names[idx] for idx in classes_present]
        if config.task == "breed"
        else [metadata.species_names[idx] for idx in classes_present]
    )
    return {
        "metadata": metadata,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "label_names": label_names,
        "num_classes": len(label_names),
        "all_trainval_samples": trainval_samples,
    }


def _collate_with_metadata(batch: list):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata = [item[2] for item in batch]
    return images, labels, metadata


def build_loaders(config: ExperimentConfig, datasets: dict[str, object]) -> dict[str, DataLoader]:
    ensure_dir(config.run_dir())
    train_loader = DataLoader(
        datasets["train_dataset"],
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate_with_metadata,
    )
    test_loader = DataLoader(
        datasets["test_dataset"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate_with_metadata,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}
