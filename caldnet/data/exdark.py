from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ExDarkDataset(Dataset):
    """
    ExDark object-detection dataset loader.

    Expected directory structure:

        data/exdark/
        ├── images/
        │   ├── Bicycle/
        │   ├── Boat/
        │   ├── ...
        │   └── Table/
        │
        └── annotations/
            ├── Bicycle/
            ├── Boat/
            ├── ...
            └── Table/

    Image files:
        .jpg
        .jpeg
        .png

    Annotation files:
        <image_name>.txt

    Example annotation:

        % bbGt version=3
        Cat 124 150 253 75 0 0 0 0 0 0 0
        Chair 31 31 361 231 0 0 0 0 0 0 0

    The first five meaningful fields are:

        class_name
        left
        top
        width
        height

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

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    def __init__(
        self,
        image_dir,
        annotation_dir,
        image_size=640,
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

        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid ExDark image/annotation pairs found."
            )

    def _build_samples(self):
        """
        Build image/annotation pairs.

        The relative path of the annotation determines
        the corresponding image.

        Example:

            annotations/Cat/2015_03042.jpg.txt

        maps to:

            images/Cat/2015_03042.jpg
        """

        samples = []

        annotation_files = sorted(
            self.annotation_dir.rglob("*.txt")
        )

        for annotation_path in annotation_files:

            relative_annotation = (
                annotation_path.relative_to(
                    self.annotation_dir
                )
            )

            image_relative_path = Path(
                str(relative_annotation)[:-4]
            )

            image_path = (
                self.image_dir
                / image_relative_path
            )

            if not image_path.exists():
                continue

            if (
                image_path.suffix.lower()
                not in self.IMAGE_EXTENSIONS
            ):
                continue

            samples.append(
                {
                    "image_path": image_path,
                    "annotation_path": annotation_path,
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