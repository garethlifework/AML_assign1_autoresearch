from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig
from .data import build_datasets, build_loaders
from .engine import ExperimentResult, run_experiment
from .models import build_model
from .utils import dump_json, get_device


@dataclass(slots=True)
class HierarchicalExperimentResult:
    router_result: ExperimentResult
    cat_result: ExperimentResult
    dog_result: ExperimentResult
    val_metrics: dict
    test_metrics: dict | None


def _load_model_for_config(config: ExperimentConfig, num_classes: int):
    device = get_device()
    model = build_model(config.backbone, num_classes).to(device)
    payload = torch.load(config.source_checkpoint_path(), map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, device


def _stitch_logits(local_probs: torch.Tensor, local_to_global: dict[int, int], num_global_classes: int, device: torch.device) -> torch.Tensor:
    stitched = torch.zeros(local_probs.size(0), num_global_classes, device=device)
    for local_idx, global_idx in local_to_global.items():
        stitched[:, global_idx] = local_probs[:, local_idx]
    return stitched


def _evaluate_hierarchy(router_config: ExperimentConfig, cat_config: ExperimentConfig, dog_config: ExperimentConfig, split_name: str) -> dict:
    breed_eval_config = ExperimentConfig.from_dict(cat_config.to_dict())
    breed_eval_config.task = "breed"
    breed_eval_config.species_filter = None
    breed_eval_config.evaluate_test = split_name == "test"
    breed_eval_config.resume_if_available = True
    datasets = build_datasets(breed_eval_config)
    loaders = build_loaders(breed_eval_config, datasets)
    loader = loaders[split_name]

    router_model, device = _load_model_for_config(router_config, 2)
    cat_model, _ = _load_model_for_config(cat_config, len(build_datasets(cat_config)["label_names"]))
    dog_model, _ = _load_model_for_config(dog_config, len(build_datasets(dog_config)["label_names"]))

    cat_datasets = build_datasets(cat_config)
    dog_datasets = build_datasets(dog_config)
    cat_mapping = cat_datasets["test_dataset"].local_to_global
    dog_mapping = dog_datasets["test_dataset"].local_to_global
    num_global_classes = len(datasets["label_names"])

    hard_correct = 0
    soft_correct = 0
    oracle_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            labels = labels.to(device)
            species_true = torch.tensor([item["species_idx"] for item in metadata], device=device)

            router_probs = torch.softmax(router_model(images), dim=1)
            cat_probs = torch.softmax(cat_model(images), dim=1)
            dog_probs = torch.softmax(dog_model(images), dim=1)

            cat_global = _stitch_logits(cat_probs, cat_mapping, num_global_classes, device)
            dog_global = _stitch_logits(dog_probs, dog_mapping, num_global_classes, device)

            hard_species = router_probs.argmax(dim=1)
            hard_logits = torch.where(hard_species.unsqueeze(1) == 0, cat_global, dog_global)
            soft_logits = router_probs[:, 0:1] * cat_global + router_probs[:, 1:2] * dog_global
            oracle_logits = torch.where(species_true.unsqueeze(1) == 0, cat_global, dog_global)

            hard_correct += (hard_logits.argmax(dim=1) == labels).sum().item()
            soft_correct += (soft_logits.argmax(dim=1) == labels).sum().item()
            oracle_correct += (oracle_logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return {
        "hard_route_acc": hard_correct / total,
        "soft_route_acc": soft_correct / total,
        "oracle_route_acc": oracle_correct / total,
        "total_examples": total,
    }


def run_hierarchical_experiment(
    router_config: ExperimentConfig,
    cat_config: ExperimentConfig,
    dog_config: ExperimentConfig,
) -> HierarchicalExperimentResult:
    router_result = run_experiment(router_config)
    cat_result = run_experiment(cat_config)
    dog_result = run_experiment(dog_config)

    val_metrics = _evaluate_hierarchy(router_config, cat_config, dog_config, split_name="val")
    test_metrics = None
    if router_config.evaluate_test or cat_config.evaluate_test or dog_config.evaluate_test:
        test_metrics = _evaluate_hierarchy(router_config, cat_config, dog_config, split_name="test")

    summary = {
        "router": router_result.config.to_dict(),
        "cat": cat_result.config.to_dict(),
        "dog": dog_result.config.to_dict(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    dump_json(router_config.paths.output_path() / f"{router_config.name}__hierarchy_summary.json", summary)
    return HierarchicalExperimentResult(
        router_result=router_result,
        cat_result=cat_result,
        dog_result=dog_result,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
