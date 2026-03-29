from __future__ import annotations

import tempfile
import unittest

from pet_researcher import ExperimentConfig
from pet_researcher.autonomous import (
    AutonomousSettings,
    _flat_candidates,
    _focused_candidates,
    _leaderboard_path,
    _write_leaderboard,
)


class AutonomousTests(unittest.TestCase):
    def test_flat_candidates_present(self):
        names = [cfg.name for cfg in _flat_candidates()]
        self.assertIn("auto_r50_layer4_raw", names)
        self.assertIn("auto_r50_layer3_4_bbox", names)

    def test_leaderboard_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = AutonomousSettings(output_dir=tmp)
            _write_leaderboard(tmp, [{"name": "a", "val_acc": 0.9, "config": {}, "run_dir": "runs/a", "checkpoint_path": "runs/a/best_model.pt", "val_macro_recall": 0.8, "test_acc": None}], settings)
            self.assertTrue(_leaderboard_path(tmp).exists())

    def test_focused_candidates_expand_winner(self):
        best = ExperimentConfig(
            name="auto_r50_full_raw_trimap",
            backbone="resnet50",
            image_mode="trimap_foreground",
            augmentation="mild",
            fine_tune_scope="full",
            finetune_lr=3e-5,
            finetune_epochs=8,
        )
        candidates = _focused_candidates(best)
        names = [cfg.name for cfg in candidates]
        self.assertIn("auto_r50_full_raw_trimap_img256_ep12_lr3e5", names)
        self.assertIn("auto_r50_full_raw_trimap_img320_ep16_lr1e5", names)


if __name__ == "__main__":
    unittest.main()
