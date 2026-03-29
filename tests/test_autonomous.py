from __future__ import annotations

import tempfile
import unittest

from pet_researcher.autonomous import AutonomousSettings, _flat_candidates, _leaderboard_path, _write_leaderboard


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


if __name__ == "__main__":
    unittest.main()
