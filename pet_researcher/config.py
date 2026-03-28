from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

TaskName = Literal["breed", "species"]
BackboneName = Literal["resnet18", "resnet50"]
ImageMode = Literal["raw", "bbox", "trimap_foreground"]
AugmentationRecipe = Literal["none", "mild", "mild_erasing", "mixup", "cutmix"]
FineTuneScope = Literal["fc", "layer4", "layer3_4", "full"]
SchedulerName = Literal["none", "cosine", "onecycle"]


@dataclass(slots=True)
class PathsConfig:
    data_dir: str = "./data/oxford-iiit-pet"
    output_dir: str = "./runs"

    def data_path(self) -> Path:
        return Path(self.data_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    task: TaskName = "breed"
    backbone: BackboneName = "resnet18"
    image_mode: ImageMode = "raw"
    augmentation: AugmentationRecipe = "none"
    fine_tune_scope: FineTuneScope = "layer3_4"
    optimizer: Literal["adamw"] = "adamw"
    scheduler: SchedulerName = "cosine"
    image_size: int = 224
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 0
    val_fraction: float = 0.2
    seed: int = 42
    split_seed: int = 42
    head_epochs: int = 2
    finetune_epochs: int = 8
    patience: int = 3
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    use_amp: bool = True
    tta: bool = False
    tta_passes: int = 4
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    random_erasing_p: float = 0.0
    species_filter: int | None = None
    source_run_name: str | None = None
    save_predictions: bool = True
    save_confusion: bool = True
    evaluate_test: bool = False
    resume_if_available: bool = False
    paths: PathsConfig = field(default_factory=PathsConfig)

    def run_dir(self) -> Path:
        return self.paths.output_path() / self.name

    def checkpoint_path(self) -> Path:
        return self.run_dir() / "best_model.pt"

    def source_checkpoint_path(self) -> Path:
        if self.source_run_name:
            return self.paths.output_path() / self.source_run_name / "best_model.pt"
        return self.checkpoint_path()

    def summary_path(self) -> Path:
        return self.run_dir() / "metrics.json"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["paths"]["data_dir"] = str(Path(data["paths"]["data_dir"]))
        data["paths"]["output_dir"] = str(Path(data["paths"]["output_dir"]))
        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "ExperimentConfig":
        payload = dict(payload)
        paths = payload.get("paths", {})
        payload["paths"] = PathsConfig(
            data_dir=paths.get("data_dir", "./data/oxford-iiit-pet"),
            output_dir=paths.get("output_dir", "./runs"),
        )
        return cls(**payload)
