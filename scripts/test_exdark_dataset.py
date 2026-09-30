from pathlib import Path

import torch

from caldnet.data.exdark import (
    ExDarkDataset,
)


def inspect_dataset(
    dataset,
    split_name,
):
    print()
    print("=" * 60)
    print(split_name)
    print("=" * 60)

    print(
        "Dataset size:",
        len(dataset),
    )

    image, targets = dataset[0]

    print(
        "Image shape:",
        tuple(image.shape),
    )

    print(
        "Image dtype:",
        image.dtype,
    )

    print(
        "Image range:",
        float(image.min()),
        "to",
        float(image.max()),
    )

    print(
        "Target shape:",
        tuple(targets.shape),
    )

    print(
        "Targets:"
    )

    print(targets)

    if targets.numel() > 0:

        coordinates = targets[:, :4]
        class_ids = targets[:, 4].long()

        print(
            "Coordinate minimum:",
            float(coordinates.min()),
        )

        print(
            "Coordinate maximum:",
            float(coordinates.max()),
        )

        print(
            "Class IDs:",
            sorted(
                set(
                    class_ids.tolist()
                )
            ),
        )

        print(
            "Number of objects:",
            targets.shape[0],
        )

        assert torch.all(
            coordinates >= 0.0
        )

        assert torch.all(
            coordinates <= 1.0
        )

        assert torch.all(
            class_ids >= 0
        )

        assert torch.all(
            class_ids < 12
        )

        print(
            "[PASS] Target values are valid."
        )


def main():

    dataset_root = Path(
        "data/exdark"
    )

    image_dir = (
        dataset_root
        / "images"
    )

    annotation_dir = (
        dataset_root
        / "annotations"
    )

    print(
        "Dataset root:",
        dataset_root,
    )

    print(
        "Image directory:",
        image_dir,
    )

    print(
        "Annotation directory:",
        annotation_dir,
    )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Missing image directory: "
            f"{image_dir}"
        )

    if not annotation_dir.exists():
        raise FileNotFoundError(
            f"Missing annotation directory: "
            f"{annotation_dir}"
        )

    dataset = ExDarkDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        image_size=640,
    )

    print()
    print(
        "Total paired samples:",
        len(dataset),
    )

    assert len(dataset) == 7363, (
        "Expected 7363 paired samples, "
        f"found {len(dataset)}."
    )

    print(
        "[PASS] Expected 7363 samples found."
    )

    inspect_dataset(
        dataset,
        "EXDARK DATASET",
    )

    print()
    print("=" * 60)
    print(
        "ExDark dataset loader test complete."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()