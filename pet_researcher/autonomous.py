from __future__ import annotations

from dataclasses import asdict
import time
from dataclasses import dataclass
from pathlib import Path

from .config import ExperimentConfig
from .engine import final_test_report, load_experiment_result, run_experiment
from .utils import dump_json


@dataclass(slots=True)
class AutonomousSettings:
    target_val_acc: float = 0.95
    max_experiments: int = 8
    top_k_finalists: int = 2
    include_hierarchy: bool = False
    output_dir: str = "./runs"


def _leaderboard_path(output_dir: str) -> Path:
    return Path(output_dir) / "leaderboard.json"


def _state_path(output_dir: str) -> Path:
    return Path(output_dir) / "autonomous_state.json"


def _candidate_name(name: str) -> str:
    return name


def _flat_candidates() -> list[ExperimentConfig]:
    shared = dict(
        backbone="resnet50",
        scheduler="cosine",
        head_epochs=2,
        finetune_epochs=8,
        head_lr=1e-3,
        weight_decay=1e-4,
        train_batch_size=16,
        eval_batch_size=32,
        resume_if_available=True,
    )
    return [
        ExperimentConfig(name="auto_r50_layer4_raw", augmentation="mild", fine_tune_scope="layer4", finetune_lr=1e-4, label_smoothing=0.05, **shared),
        ExperimentConfig(name="auto_r50_layer3_4_raw", augmentation="mild", fine_tune_scope="layer3_4", finetune_lr=1e-4, label_smoothing=0.05, **shared),
        ExperimentConfig(name="auto_r50_full_raw", augmentation="mild", fine_tune_scope="full", finetune_lr=3e-5, label_smoothing=0.05, **shared),
        ExperimentConfig(name="auto_r50_layer4_bbox", image_mode="bbox", augmentation="mild", fine_tune_scope="layer4", finetune_lr=1e-4, label_smoothing=0.05, **shared),
        ExperimentConfig(name="auto_r50_layer3_4_bbox", image_mode="bbox", augmentation="mild", fine_tune_scope="layer3_4", finetune_lr=1e-4, label_smoothing=0.05, **shared),
        ExperimentConfig(name="auto_r50_layer3_4_erasing", augmentation="mild_erasing", fine_tune_scope="layer3_4", finetune_lr=1e-4, random_erasing_p=0.10, label_smoothing=0.05, **shared),
        ExperimentConfig(name="auto_r50_layer3_4_raw_ls01", augmentation="mild", fine_tune_scope="layer3_4", finetune_lr=1e-4, label_smoothing=0.10, **shared),
    ]


def _trimap_candidates(best_raw: ExperimentConfig | None) -> list[ExperimentConfig]:
    if best_raw is None:
        return []
    base = ExperimentConfig.from_dict(best_raw.to_dict())
    base.resume_if_available = True
    return [
        ExperimentConfig.from_dict({**base.to_dict(), "name": f"{base.name}_trimap", "image_mode": "trimap_foreground"}),
        ExperimentConfig.from_dict({**base.to_dict(), "name": f"{base.name}_bbox_full", "image_mode": "bbox", "fine_tune_scope": "full", "finetune_lr": 3e-5}),
    ]


def _focused_candidates(best_existing: ExperimentConfig | None) -> list[ExperimentConfig]:
    if best_existing is None:
        return []

    base_payload = best_existing.to_dict()
    candidates = []
    variants = [
        {
            "suffix": "img256_ep12_lr3e5",
            "image_size": 256,
            "finetune_epochs": 12,
            "finetune_lr": 3e-5,
            "train_batch_size": 16,
            "eval_batch_size": 32,
        },
        {
            "suffix": "img256_ep16_lr1e5",
            "image_size": 256,
            "finetune_epochs": 16,
            "finetune_lr": 1e-5,
            "train_batch_size": 16,
            "eval_batch_size": 32,
        },
        {
            "suffix": "img320_ep12_lr3e5",
            "image_size": 320,
            "finetune_epochs": 12,
            "finetune_lr": 3e-5,
            "train_batch_size": 8,
            "eval_batch_size": 16,
        },
        {
            "suffix": "img320_ep16_lr1e5",
            "image_size": 320,
            "finetune_epochs": 16,
            "finetune_lr": 1e-5,
            "train_batch_size": 8,
            "eval_batch_size": 16,
        },
    ]

    for variant in variants:
        payload = {
            **base_payload,
            **variant,
            "name": f"{best_existing.name}_{variant['suffix']}",
            "resume_if_available": True,
        }
        payload.pop("suffix", None)
        candidates.append(ExperimentConfig.from_dict(payload))

    return candidates


def _finalist_configs(results: list[dict], top_k: int) -> list[ExperimentConfig]:
    finalists = []
    for row in sorted(results, key=lambda item: item["val_acc"], reverse=True)[:top_k]:
        cfg = ExperimentConfig.from_dict(row["config"])
        cfg.name = f"{cfg.name}_tta"
        cfg.tta = True
        cfg.source_run_name = row["name"]
        cfg.resume_if_available = True
        finalists.append(cfg)
    return finalists


def _read_leaderboard(output_dir: str) -> list[dict]:
    path = _leaderboard_path(output_dir)
    if not path.exists():
        return []
    import json

    payload = json.loads(path.read_text())
    return payload.get("experiments", [])


def _write_leaderboard(output_dir: str, experiments: list[dict], settings: AutonomousSettings) -> None:
    payload = {
        "settings": asdict(settings),
        "experiments": sorted(experiments, key=lambda item: item["val_acc"], reverse=True),
        "updated_at": time.time(),
    }
    dump_json(_leaderboard_path(output_dir), payload)


def _result_row(result) -> dict:
    return {
        "name": result.config.name,
        "val_acc": result.val_metrics["acc"],
        "val_macro_recall": result.val_metrics.get("macro_recall"),
        "test_acc": result.test_metrics["acc"] if result.test_metrics else None,
        "config": result.config.to_dict(),
        "run_dir": result.run_dir,
        "checkpoint_path": result.checkpoint_path,
    }


def _already_done(name: str, experiments: list[dict]) -> bool:
    return any(row["name"] == name for row in experiments)


def _best_non_tta(experiments: list[dict]) -> ExperimentConfig | None:
    eligible = [row for row in experiments if "_tta" not in row["name"]]
    if not eligible:
        return None
    return ExperimentConfig.from_dict(sorted(eligible, key=lambda item: item["val_acc"], reverse=True)[0]["config"])


def run_autonomous_loop(settings: AutonomousSettings | None = None) -> dict:
    settings = settings or AutonomousSettings()
    experiments = _read_leaderboard(settings.output_dir)
    executed_count = len(experiments)

    queue = list(_flat_candidates())
    if experiments:
        best_existing = _best_non_tta(experiments)
        queue.extend(_trimap_candidates(best_existing))
        queue.extend(_focused_candidates(best_existing))

    for config in queue:
        if executed_count >= settings.max_experiments:
            break
        if _already_done(_candidate_name(config.name), experiments):
            continue

        result = run_experiment(config)
        experiments.append(_result_row(result))
        executed_count += 1
        _write_leaderboard(settings.output_dir, experiments, settings)

        best_val = max(row["val_acc"] for row in experiments)
        if best_val >= settings.target_val_acc:
            break

    best_existing = _best_non_tta(experiments)
    if best_existing is not None:
        for candidate in _trimap_candidates(best_existing):
            if executed_count >= settings.max_experiments:
                break
            if _already_done(candidate.name, experiments):
                continue
            result = run_experiment(candidate)
            experiments.append(_result_row(result))
            executed_count += 1
            _write_leaderboard(settings.output_dir, experiments, settings)

    best_existing = _best_non_tta(experiments)
    if best_existing is not None:
        for candidate in _focused_candidates(best_existing):
            if executed_count >= settings.max_experiments:
                break
            if _already_done(candidate.name, experiments):
                continue
            result = run_experiment(candidate)
            experiments.append(_result_row(result))
            executed_count += 1
            _write_leaderboard(settings.output_dir, experiments, settings)

    finalists = _finalist_configs(experiments, settings.top_k_finalists)
    final_report = final_test_report(finalists) if finalists else {"models": [], "best_model": None}
    state = {
        "status": "completed",
        "experiments_run": len(experiments),
        "best_val_acc": max((row["val_acc"] for row in experiments), default=None),
        "best_model_name": final_report.get("best_model", {}).get("name") if final_report.get("best_model") else None,
        "final_report_path": str(Path(settings.output_dir) / "final_test_report.json"),
        "leaderboard_path": str(_leaderboard_path(settings.output_dir)),
        "updated_at": time.time(),
    }
    dump_json(_state_path(settings.output_dir), state)
    return {"leaderboard": experiments, "final_report": final_report, "state": state}
