from __future__ import annotations

from pet_researcher import ExperimentConfig, run_experiment


def configs() -> list[ExperimentConfig]:
    common = dict(
        backbone="resnet18",
        head_epochs=2,
        finetune_epochs=6,
        val_fraction=0.2,
        split_seed=42,
        seed=42,
    )
    return [
        ExperimentConfig(
            name="sweep_resnet18_control",
            augmentation="none",
            fine_tune_scope="layer3_4",
            scheduler="none",
            label_smoothing=0.0,
            finetune_lr=1e-4,
            **common,
        ),
        ExperimentConfig(
            name="sweep_resnet18_cosine_ls",
            augmentation="none",
            fine_tune_scope="layer3_4",
            scheduler="cosine",
            label_smoothing=0.05,
            finetune_lr=1e-4,
            **common,
        ),
        ExperimentConfig(
            name="sweep_resnet18_mild_aug",
            augmentation="mild",
            fine_tune_scope="layer3_4",
            scheduler="cosine",
            label_smoothing=0.05,
            finetune_lr=2e-4,
            **common,
        ),
        ExperimentConfig(
            name="sweep_resnet18_layer4_mild_aug",
            augmentation="mild",
            fine_tune_scope="layer4",
            scheduler="cosine",
            label_smoothing=0.05,
            finetune_lr=3e-4,
            **common,
        ),
        ExperimentConfig(
            name="sweep_resnet18_erasing",
            augmentation="mild_erasing",
            fine_tune_scope="layer3_4",
            scheduler="cosine",
            label_smoothing=0.05,
            finetune_lr=2e-4,
            random_erasing_p=0.10,
            **common,
        ),
    ]


def main() -> None:
    results = []
    for cfg in configs():
        print(f"=== Running {cfg.name} ===")
        result = run_experiment(cfg)
        print(f"{cfg.name}: {result.val_metrics}")
        results.append((cfg.name, result.val_metrics["acc"], result.run_dir))

    print("=== Sweep summary ===")
    for name, acc, run_dir in sorted(results, key=lambda item: item[1], reverse=True):
        print(f"{name}: val_acc={acc:.4f} run_dir={run_dir}")


if __name__ == "__main__":
    main()
