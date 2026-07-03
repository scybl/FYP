import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile as tiff

from LoadData.data import get_dataset


def _base_config(root):
    augmentations = {
        "geometric_transforms": [
            {"type": "Resize", "params": {"size": [16, 16]}},
        ],
        "color_transforms": [],
    }
    return {
        "data_loader": {"batch_size": 2, "shuffle": True, "num_workers": 0},
        "datasets": {
            "isic2018": {
                "class_num": 1,
                "in_channel": 3,
                "dataset_path": str(root),
                "img_prefix": "ISIC_",
                "seg_prefix": "ISIC_",
                "img_suffix": ".jpg",
                "seg_suffix": "_segmentation.png",
                "train_img": "isic2018/train/img",
                "train_mask": "isic2018/train/mask",
                "val_img": "isic2018/val/img",
                "val_mask": "isic2018/val/mask",
                "test_img": "isic2018/test/img",
                "test_mask": "isic2018/test/mask",
                "augmentations": augmentations,
            },
            "kvasir": {
                "class_num": 1,
                "in_channel": 3,
                "dataset_path": str(root),
                "train_img": "Kvasir-SEG/train/img",
                "train_mask": "Kvasir-SEG/train/mask",
                "val_img": "Kvasir-SEG/val/img",
                "val_mask": "Kvasir-SEG/val/mask",
                "test_img": "Kvasir-SEG/test/img",
                "test_mask": "Kvasir-SEG/test/mask",
                "augmentations": augmentations,
            },
            "clinicdb": {
                "class_num": 1,
                "in_channel": 3,
                "dataset_path": str(root),
                "train_img": "CVC-ClinicDB/train/img",
                "train_mask": "CVC-ClinicDB/train/mask",
                "val_img": "CVC-ClinicDB/val/img",
                "val_mask": "CVC-ClinicDB/val/mask",
                "test_img": "CVC-ClinicDB/test/img",
                "test_mask": "CVC-ClinicDB/test/mask",
                "augmentations": augmentations,
            },
        },
    }


def _write_png(path, channels=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 3:
        array = np.zeros((24, 24, 3), dtype=np.uint8)
        array[4:18, 6:20] = [120, 80, 200]
    else:
        array = np.zeros((24, 24), dtype=np.uint8)
        array[5:16, 7:18] = 255
    Image.fromarray(array).save(path)


def _write_tiff(path, channels=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 3:
        array = np.zeros((24, 24, 3), dtype=np.uint8)
        array[4:18, 6:20] = [120, 80, 200]
    else:
        array = np.zeros((24, 24), dtype=np.uint8)
        array[5:16, 7:18] = 255
    tiff.imwrite(path, array)


class DataLoadingTest(unittest.TestCase):
    def test_isic_loader_uses_mask_name_to_find_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(2):
                _write_png(root / "isic2018/train/img" / f"ISIC_{index}.jpg")
                _write_png(root / "isic2018/train/mask" / f"ISIC_{index}_segmentation.png", channels=1)

            loader = get_dataset(_base_config(root), "isic2018", "train")
            images, masks = next(iter(loader))

            self.assertEqual(tuple(images.shape), (2, 3, 16, 16))
            self.assertEqual(tuple(masks.shape), (2, 1, 16, 16))

    def test_kvasir_loader_accepts_same_image_and_mask_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(2):
                _write_png(root / "Kvasir-SEG/test/img" / f"sample_{index}.png")
                _write_png(root / "Kvasir-SEG/test/mask" / f"sample_{index}.png", channels=1)

            loader = get_dataset(_base_config(root), "kvasir", "test")
            images, masks = next(iter(loader))

            self.assertEqual(tuple(images.shape), (2, 3, 16, 16))
            self.assertEqual(tuple(masks.shape), (2, 1, 16, 16))

    def test_clinicdb_loader_pairs_sorted_tiff_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(2):
                _write_tiff(root / "CVC-ClinicDB/val/img" / f"{index}.tif")
                _write_tiff(root / "CVC-ClinicDB/val/mask" / f"{index}.tif", channels=1)

            loader = get_dataset(_base_config(root), "clinicdb", "val")
            images, masks = next(iter(loader))

            self.assertEqual(tuple(images.shape), (2, 3, 16, 16))
            self.assertEqual(tuple(masks.shape), (2, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
