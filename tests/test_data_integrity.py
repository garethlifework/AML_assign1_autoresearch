from __future__ import annotations

import unittest

from pet_researcher import ExperimentConfig
from pet_researcher.data import build_datasets, load_metadata


class DataIntegrityTests(unittest.TestCase):
    def test_metadata_has_37_classes(self):
        metadata = load_metadata(ExperimentConfig(name="tmp").paths.data_path())
        self.assertEqual(len(metadata.class_names), 37)
        self.assertEqual(set(metadata.species_names), {"cat", "dog"})

    def test_stratified_split_keeps_all_classes(self):
        datasets = build_datasets(ExperimentConfig(name="tmp"))
        train_labels = {datasets["all_trainval_samples"][idx].breed_idx for idx in datasets["train_indices"]}
        val_labels = {datasets["all_trainval_samples"][idx].breed_idx for idx in datasets["val_indices"]}
        self.assertEqual(len(train_labels), 37)
        self.assertEqual(len(val_labels), 37)

    def test_species_specialists_cover_expected_breeds(self):
        cat_data = build_datasets(ExperimentConfig(name="cat_tmp", species_filter=0))
        dog_data = build_datasets(ExperimentConfig(name="dog_tmp", species_filter=1))
        self.assertEqual(cat_data["num_classes"], 12)
        self.assertEqual(dog_data["num_classes"], 25)

    def test_annotation_assets_are_present_for_sample(self):
        datasets = build_datasets(ExperimentConfig(name="tmp"))
        sample = datasets["all_trainval_samples"][0]
        self.assertTrue(sample.image_path.exists())
        self.assertTrue(sample.bbox_path.exists())
        self.assertTrue(sample.trimap_path.exists())


if __name__ == "__main__":
    unittest.main()
