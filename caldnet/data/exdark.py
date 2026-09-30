from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ExDarkDataset(Dataset):
    """
    ExDark object-detection dataset loader.

    Supports the official ExDark experiment split defined by
    imageclasslist.txt.

    Expected directory structure:

        data/exdark/
        ├── images/
        │   ├── Bicycle/
        │   ├── Boat/
        │   ├── ...
        │   └── Table/
        │
        ├── annotations/
        │   ├── Bicycle/
        │   ├── Boat/
        │   ├── ...
        │   └── Table/
        │
        └── imageclasslist.txt

    Official imageclasslist.txt format:

        Name | Class | Light | In/Out | Train/Val/Test

    Example:

        2015_00001.png 1 2 1 1
        2015_00002.png 1 6 2 1

    Split values:

        1 = train
        2 = val
        3 = test

    Image annotations use:

        class_name left top width height ...

    Output target format:

        [x_center, y_center, width, height, class_id]

    Coordinates are normalized to [0, 1].
    """

    CLASS_NAMES = [
        "Bicycle",
        "Boat",
        "Bottle",
        "Bus",
        "Car",
        "Cat",
        "Chair",
        "Cup",
        "Dog",
        "Motorbike",
        "People",
        "Table",
    ]

    CLASS_TO_ID = {
        class_name: class_id
        for class_id, class_name in enumerate(CLASS_NAMES)
    }

    OFFICIAL_CLASS_TO_ID = {
        class_id + 1: class_id
        for class_id in range(len(CLASS_NAMES))
    }

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    SPLIT_TO_ID = {
        "train": 1,
        "val": 2,
        "test": 3,
    }

    SPLIT_ALIASES = {
        "training": "train",
        "validation": "val",
        "testing": "test",
        "train": "train",
        "val": "val",
        "test": "test",
    }

    def __init__(
        self,
        image_dir,
        annotation_dir,
        image_size=640,
        imageclasslist_path=None,
        split=None,
    ):
        super().__init__()

        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: "
                f"{self.image_dir}"
            )

        if not self.annotation_dir.exists():
            raise FileNotFoundError(
                f"Annotation directory does not exist: "
                f"{self.annotation_dir}"
            )

        if split is not None:
            split = self.SPLIT_ALIASES.get(
                split.lower()
            )

            if split is None:
                raise ValueError(
                    "Invalid ExDark split. "
                    "Expected one of: "
                    "'train', 'val', 'test', "
                    "'training', 'validation', 'testing'."
                )

            if imageclasslist_path is None:
                raise ValueError(
                    "imageclasslist_path is required "
                    "when split is specified."
                )

        self.split = split

        self.imageclasslist_path = (
            Path(imageclasslist_path)
            if imageclasslist_path is not None
            else None
        )

        if self.imageclasslist_path is not None:
            if not self.imageclasslist_path.exists():
                raise FileNotFoundError(
                    "ExDark imageclasslist.txt does not exist: "
                    f"{self.imageclasslist_path}"
                )

        self.split_metadata = {}

        if self.imageclasslist_path is not None:
            self.split_metadata = (
                self._parse_imageclasslist()
            )

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            if self.split is None:
                raise RuntimeError(
                    "No valid ExDark image/annotation pairs found."
                )

            raise RuntimeError(
                "No valid ExDark image/annotation pairs "
                f"found for split '{self.split}'."
            )

    def _parse_imageclasslist(self):
        """
        Parse the official ExDark imageclasslist.txt.

        Returns:

            {
                "filename.jpg": {
                    "class_id": 0,
                    "light_id": 1,
                    "indoor_outdoor_id": 1,
                    "split_id": 1,
                }
            }
        """

        metadata = {}

        with self.imageclasslist_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            lines = file.readlines()

        for line_number, raw_line in enumerate(
            lines,
            start=1,
        ):

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith("Name"):
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            image_name = parts[0]

            try:
                class_id = int(parts[1])
                light_id = int(parts[2])
                indoor_outdoor_id = int(parts[3])
                split_id = int(parts[4])
            except ValueError as error:
                raise ValueError(
                    "Invalid ExDark imageclasslist entry "
                    f"at line {line_number}: {line}"
                ) from error

            if class_id not in self.OFFICIAL_CLASS_TO_ID:
                raise ValueError(
                    "Invalid ExDark class ID "
                    f"{class_id} at line {line_number}."
                )

            if light_id < 1 or light_id > 10:
                raise ValueError(
                    "Invalid ExDark lighting ID "
                    f"{light_id} at line {line_number}."
                )

            if indoor_outdoor_id not in {1, 2}:
                raise ValueError(
                    "Invalid ExDark indoor/outdoor ID "
                    f"{indoor_outdoor_id} at line {line_number}."
                )

            if split_id not in {1, 2, 3}:
                raise ValueError(
                    "Invalid ExDark split ID "
                    f"{split_id} at line {line_number}."
                )

            key = image_name.lower()

            if key in metadata:
                raise ValueError(
                    "Duplicate image name in "
                    "imageclasslist.txt: "
                    f"{image_name}"
                )

            metadata[key] = {
                "image_name": image_name,
                "class_id": (
                    self.OFFICIAL_CLASS_TO_ID[class_id]
                ),
                "light_id": light_id,
                "indoor_outdoor_id": indoor_outdoor_id,
                "split_id": split_id,
            }

        if len(metadata) == 0:
            raise RuntimeError(
                "No valid entries found in "
                f"{self.imageclasslist_path}"
            )

        return metadata

    def _build_image_index(self):
        """
        Build a case-insensitive image filename index.

        Returns:

            {
                "2015_00001.jpg": Path(...),
                ...
            }

        The image filename must be unique across the local
        ExDark image directory.
        """

        image_index = {}

        for image_path in self.image_dir.rglob("*"):

            if not image_path.is_file():
                continue

            if (
                image_path.suffix.lower()
                not in self.IMAGE_EXTENSIONS
            ):
                continue

            key = image_path.name.lower()

            if key in image_index:
                raise RuntimeError(
                    "Duplicate image filename found in "
                    "ExDark image directory: "
                    f"{image_path.name}\n"
                    f"Existing path: {image_index[key]}\n"
                    f"Duplicate path: {image_path}"
                )

            image_index[key] = image_path

        return image_index

    def _build_samples(self):
        """
        Build image/annotation pairs.

        Without a split:

            All local image/annotation pairs are returned.

        With a split:

            Only images listed in the official
            imageclasslist.txt for that split are returned.
        """

        image_index = self._build_image_index()

        samples = []

        annotation_files = sorted(
            self.annotation_dir.rglob("*.txt")
        )

        annotation_index = {}

        for annotation_path in annotation_files:

            key = annotation_path.stem.lower()

            if key in annotation_index:
                raise RuntimeError(
                    "Duplicate annotation filename found: "
                    f"{annotation_path.stem}"
                )

            annotation_index[key] = annotation_path

        for image_key, image_path in sorted(
            image_index.items()
        ):

            annotation_key = image_path.name.lower()

            annotation_path = annotation_index.get(
                annotation_key
            )

            if annotation_path is None:

                annotation_key = (
                    image_path.stem.lower()
                )

                annotation_path = annotation_index.get(
                    annotation_key
                )

            if annotation_path is None:
                continue

            metadata = self.split_metadata.get(
                image_key
            )

            if self.split is not None:

                if metadata is None:
                    continue

                expected_split_id = self.SPLIT_TO_ID[
                    self.split
                ]

                if metadata["split_id"] != (
                    expected_split_id
                ):
                    continue

                expected_class_id = metadata[
                    "class_id"
                ]

                folder_name = (
                    image_path.parent.name
                )

                if folder_name in self.CLASS_TO_ID:

                    actual_folder_class_id = (
                        self.CLASS_TO_ID[
                            folder_name
                        ]
                    )

                    if (
                        actual_folder_class_id
                        != expected_class_id
                    ):
                        raise RuntimeError(
                            "ExDark class mismatch for "
                            f"{image_path.name}: "
                            f"imageclasslist class="
                            f"{self.CLASS_NAMES[expected_class_id]}, "
                            f"folder class="
                            f"{folder_name}"
                        )

            samples.append(
                {
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                    "metadata": metadata,
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def _parse_annotation(
        self,
        annotation_path,
        image_width,
        image_height,
    ):
        """
        Parse one ExDark annotation file.

        Returns:

            tensor [N, 5]

        where each row is:

            [x_center, y_center,
             width, height, class_id]
        """

        targets = []

        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            lines = file.readlines()

        for line_number, raw_line in enumerate(
            lines,
            start=1,
        ):

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("%"):
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            class_name = parts[0]

            if class_name not in self.CLASS_TO_ID:
                raise ValueError(
                    "Unknown ExDark class "
                    f"'{class_name}' in "
                    f"{annotation_path} "
                    f"at line {line_number}."
                )

            try:
                left = float(parts[1])
                top = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError as error:
                raise ValueError(
                    "Invalid bounding-box values in "
                    f"{annotation_path} "
                    f"at line {line_number}: "
                    f"{line}"
                ) from error

            if width <= 0 or height <= 0:
                continue

            x_center = (
                left + width / 2.0
            )

            y_center = (
                top + height / 2.0
            )

            x_center /= image_width
            y_center /= image_height

            width /= image_width
            height /= image_height

            x_center = np.clip(
                x_center,
                0.0,
                1.0,
            )

            y_center = np.clip(
                y_center,
                0.0,
                1.0,
            )

            width = np.clip(
                width,
                0.0,
                1.0,
            )

            height = np.clip(
                height,
                0.0,
                1.0,
            )

            class_id = self.CLASS_TO_ID[
                class_name
            ]

            targets.append(
                [
                    x_center,
                    y_center,
                    width,
                    height,
                    class_id,
                ]
            )

        if len(targets) == 0:
            return torch.empty(
                (0, 5),
                dtype=torch.float32,
            )

        return torch.tensor(
            targets,
            dtype=torch.float32,
        )

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = sample["image_path"]
        annotation_path = sample[
            "annotation_path"
        ]

        image = Image.open(
            image_path
        ).convert("RGB")

        original_width, original_height = (
            image.size
        )

        targets = self._parse_annotation(
            annotation_path=annotation_path,
            image_width=original_width,
            image_height=original_height,
        )

        image = image.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array /= 255.0

        image_tensor = torch.from_numpy(
            image_array
        )

        image_tensor = image_tensor.permute(
            2,
            0,
            1,
        ).contiguous()

        return image_tensor, targets