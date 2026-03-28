from __future__ import annotations

from pet_researcher import ExperimentConfig, final_test_report, run_experiment, run_hierarchical_experiment


def main() -> None:
    baseline = ExperimentConfig(
        name="stage1_resnet18_baseline",
        backbone="resnet18",
        augmentation="none",
        fine_tune_scope="layer3_4",
        label_smoothing=0.0,
    )
    improved = ExperimentConfig(
        name="stage2_resnet18_mild_aug",
        backbone="resnet18",
        augmentation="mild",
        fine_tune_scope="layer3_4",
        label_smoothing=0.05,
        scheduler="cosine",
        finetune_lr=3e-4,
    )
    resnet50 = ExperimentConfig(
        name="stage3_resnet50_layer4",
        backbone="resnet50",
        augmentation="mild",
        fine_tune_scope="layer4",
        label_smoothing=0.05,
        scheduler="cosine",
        finetune_lr=1e-4,
    )

    results = [run_experiment(cfg) for cfg in (baseline, improved, resnet50)]

    router = ExperimentConfig(
        name="stage4_router_species",
        task="species",
        backbone="resnet18",
        augmentation="mild",
        fine_tune_scope="layer4",
        label_smoothing=0.05,
    )
    cat = ExperimentConfig(
        name="stage4_cat_specialist",
        task="breed",
        backbone="resnet18",
        augmentation="mild",
        fine_tune_scope="layer3_4",
        label_smoothing=0.05,
        species_filter=0,
    )
    dog = ExperimentConfig(
        name="stage4_dog_specialist",
        task="breed",
        backbone="resnet18",
        augmentation="mild",
        fine_tune_scope="layer3_4",
        label_smoothing=0.05,
        species_filter=1,
    )
    run_hierarchical_experiment(router, cat, dog)

    finalists = []
    for result in sorted(results, key=lambda item: item.val_metrics["acc"], reverse=True)[:2]:
        cfg = ExperimentConfig.from_dict(result.config.to_dict())
        cfg.tta = True
        cfg.name = f"{cfg.name}_tta"
        cfg.source_run_name = result.config.name
        cfg.resume_if_available = True
        finalists.append(cfg)
    final_test_report(finalists)


if __name__ == "__main__":
    main()
