from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score
from torch import nn

from .config import ExperimentConfig
from .data import build_datasets, build_loaders
from .models import build_model, set_fine_tune_scope
from .reporting import save_confusion_matrix, save_predictions_csv
from .utils import dump_json, get_device, set_seed


@dataclass(slots=True)
class ExperimentResult:
    config: ExperimentConfig
    checkpoint_path: str
    val_metrics: dict
    test_metrics: dict | None
    label_names: list[str]
    run_dir: str


def load_experiment_result(config: ExperimentConfig) -> ExperimentResult | None:
    summary_path = config.summary_path()
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text())
    saved_config = ExperimentConfig.from_dict(payload["config"])
    return ExperimentResult(
        config=saved_config,
        checkpoint_path=str(saved_config.checkpoint_path()),
        val_metrics=payload["val_metrics"],
        test_metrics=payload.get("test_metrics"),
        label_names=payload.get("label_names", []),
        run_dir=str(saved_config.run_dir()),
    )


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = -1.0
        self.bad_epochs = 0

    def step(self, current: float) -> bool:
        if current > self.best:
            self.best = current
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience


def _soft_target_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def _mixup_or_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    mixup_alpha: float,
    cutmix_alpha: float,
):
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return images, labels, None

    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    mixed_targets = one_hot.clone()

    if cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        h, w = images.size(2), images.size(3)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, h)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = lam * images + (1.0 - lam) * images[perm]

    mixed_targets = lam * one_hot + (1.0 - lam) * one_hot[perm]
    return images, labels, mixed_targets


def _build_criterion(config: ExperimentConfig) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)


def _build_scheduler(config: ExperimentConfig, optimizer, total_steps: int):
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    if config.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=max(1, total_steps),
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def _supports_amp(device: torch.device) -> bool:
    return device.type == "cuda"


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimiser=None,
    scheduler=None,
    use_amp: bool = False,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    num_classes: int | None = None,
) -> dict:
    training = optimiser is not None
    model.train(training)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if training and use_amp else None

    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch in loader:
        images, labels = batch[:2]
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimiser.zero_grad(set_to_none=True)
            images, labels, soft_targets = _mixup_or_cutmix(
                images,
                labels,
                num_classes=num_classes or model.fc.out_features,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
            )
        else:
            soft_targets = None

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels) if soft_targets is None else _soft_target_loss(logits, soft_targets)

            if training and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            elif training:
                loss.backward()
                optimiser.step()

            if training and scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    if training and scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler.step()

    return {"loss": running_loss / total, "acc": running_correct / total}


def _predict(model: nn.Module, loader, device: torch.device, tta: bool = False, tta_passes: int = 4) -> dict:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_probs: list[list[float]] = []
    all_meta: list[dict] = []

    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            logits = model(images)
            if tta:
                tta_logits = [logits]
                for idx in range(max(0, tta_passes - 1)):
                    augmented = images.flip(-1) if idx % 2 == 0 else images
                    tta_logits.append(model(augmented))
                logits = torch.stack(tta_logits, dim=0).mean(dim=0)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_true.extend(labels.tolist())
            all_pred.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_meta.extend(metadata)

    return {
        "y_true": all_true,
        "y_pred": all_pred,
        "probs": all_probs,
        "metadata": all_meta,
    }


def _metrics_from_predictions(prediction_bundle: dict) -> dict:
    y_true = prediction_bundle["y_true"]
    y_pred = prediction_bundle["y_pred"]
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "acc": float(np.mean(np.equal(y_true, y_pred))),
        "macro_recall": float(np.mean(per_class_recall)),
    }


def _save_prediction_artifacts(config: ExperimentConfig, prediction_bundle: dict, label_names: list[str], split_name: str) -> None:
    rows = []
    for meta, true_idx, pred_idx, probs in zip(
        prediction_bundle["metadata"],
        prediction_bundle["y_true"],
        prediction_bundle["y_pred"],
        prediction_bundle["probs"],
    ):
        rows.append(
            {
                "image_id": meta["image_id"],
                "true_label": label_names[true_idx],
                "pred_label": label_names[pred_idx],
                "true_index": true_idx,
                "pred_index": pred_idx,
                "confidence": max(probs),
            }
        )
    if config.save_predictions:
        save_predictions_csv(config.run_dir() / f"{split_name}_predictions.csv", rows)
    if config.save_confusion:
        save_confusion_matrix(
            config.run_dir() / f"{split_name}_confusion.png",
            prediction_bundle["y_true"],
            prediction_bundle["y_pred"],
            label_names,
        )


def _train_phase(
    model: nn.Module,
    config: ExperimentConfig,
    loaders: dict,
    criterion: nn.Module,
    device: torch.device,
    lr: float,
    num_epochs: int,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> tuple[nn.Module, list[dict]]:
    optimiser = _build_optimizer(model, lr=lr, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(config, optimiser, total_steps=num_epochs * len(loaders["train"]))
    history = []
    best_weights = copy.deepcopy(model.state_dict())
    best_val = -1.0
    stopper = EarlyStopper(config.patience)

    for epoch in range(num_epochs):
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimiser=optimiser,
            scheduler=scheduler,
            use_amp=config.use_amp and _supports_amp(device),
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            num_classes=model.fc.out_features,
        )
        val_metrics = _run_epoch(model, loaders["val"], criterion, device)
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(
            f"[{config.name}] epoch {epoch + 1:02d}/{num_epochs:02d} "
            f"train_acc={train_metrics['acc']:.4f} train_loss={train_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f}"
        )

        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            best_weights = copy.deepcopy(model.state_dict())
        if stopper.step(val_metrics["acc"]):
            break

    model.load_state_dict(best_weights)
    return model, history


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    set_seed(config.seed)
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.resume_if_available:
        cached_result = load_experiment_result(config)
        if cached_result is not None and (not config.evaluate_test or cached_result.test_metrics is not None):
            return cached_result

    device = get_device()
    datasets = build_datasets(config)
    loaders = build_loaders(config, datasets)
    label_names = datasets["label_names"]
    criterion = _build_criterion(config)
    model = build_model(config.backbone, datasets["num_classes"]).to(device)

    if config.resume_if_available and config.source_checkpoint_path().exists():
        payload = torch.load(config.source_checkpoint_path(), map_location=device)
        model.load_state_dict(payload["model_state"])
        history = payload.get("history", [])
    else:
        history = []
        set_fine_tune_scope(model, "fc")
        model, head_history = _train_phase(
            model,
            config,
            loaders,
            criterion,
            device,
            lr=config.head_lr,
            num_epochs=config.head_epochs,
        )
        history.extend(head_history)

        set_fine_tune_scope(model, config.fine_tune_scope)
        mixup_alpha = config.mixup_alpha if config.augmentation == "mixup" else 0.0
        cutmix_alpha = config.cutmix_alpha if config.augmentation == "cutmix" else 0.0
        model, ft_history = _train_phase(
            model,
            config,
            loaders,
            criterion,
            device,
            lr=config.finetune_lr,
            num_epochs=config.finetune_epochs,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        )
        history.extend(ft_history)
        torch.save(
            {
                "config": config.to_dict(),
                "model_state": model.state_dict(),
                "history": history,
                "label_names": label_names,
            },
            config.checkpoint_path(),
        )

    val_predictions = _predict(model, loaders["val"], device, tta=config.tta, tta_passes=config.tta_passes)
    val_metrics = _metrics_from_predictions(val_predictions)
    _save_prediction_artifacts(config, val_predictions, label_names, "val")

    test_metrics = None
    if config.evaluate_test:
        test_predictions = _predict(model, loaders["test"], device, tta=config.tta, tta_passes=config.tta_passes)
        test_metrics = _metrics_from_predictions(test_predictions)
        _save_prediction_artifacts(config, test_predictions, label_names, "test")

    summary = {
        "config": config.to_dict(),
        "device": str(device),
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "label_names": label_names,
        "timestamp": time.time(),
    }
    dump_json(config.summary_path(), summary)
    return ExperimentResult(
        config=config,
        checkpoint_path=str(config.checkpoint_path()),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        label_names=label_names,
        run_dir=str(run_dir),
    )


def final_test_report(shortlist_configs: list[ExperimentConfig]) -> dict:
    report_rows = []
    best_row = None
    for config in shortlist_configs:
        frozen = ExperimentConfig.from_dict(config.to_dict())
        frozen.evaluate_test = True
        frozen.resume_if_available = True
        result = run_experiment(frozen)
        row = {
            "name": frozen.name,
            "backbone": frozen.backbone,
            "image_mode": frozen.image_mode,
            "tta": frozen.tta,
            "val_acc": result.val_metrics["acc"],
            "test_acc": result.test_metrics["acc"] if result.test_metrics else None,
            "checkpoint_path": result.checkpoint_path,
        }
        report_rows.append(row)
        if best_row is None or (row["test_acc"] or -1.0) > (best_row["test_acc"] or -1.0):
            best_row = row

    report = {"models": report_rows, "best_model": best_row}
    if shortlist_configs:
        dump_json(shortlist_configs[0].paths.output_path() / "final_test_report.json", report)
    return report
