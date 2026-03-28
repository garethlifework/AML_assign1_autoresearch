from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from sklearn.metrics import confusion_matrix

from .utils import ensure_dir


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path: Path, y_true: list[int], y_pred: list[int], label_names: list[str]) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
