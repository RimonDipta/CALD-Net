from collections import Counter
from pathlib import Path

from caldnet.data.exdark import ExDarkDataset


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


EXPECTED_SPLIT_COUNTS = {
    "train": 3000,
    "val": 1800,
    "test": 2563,
}


EXPECTED_TRAIN_PER_CLASS = 250
EXPECTED_VAL_PER_CLASS = 150


def inspect_split(
    split_name,
    image_dir,
    annotation_dir,
    imageclasslist_path,
):
    print()
    print("=" * 70)
    print(
        f"EXDARK {split_name.upper()} SPLIT"
    )
    print("=" * 70)

    dataset = ExDarkDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        image_size=640,
        imageclasslist_path=imageclasslist_path,
        split=split_name,
    )

    print(
        "Dataset size:",
        len(dataset),
    )

    expected_count = EXPECTED_SPLIT_COUNTS[
        split_name
    ]

    assert len(dataset) == expected_count, (
        f"Expected {expected_count} images in "
        f"{split_name}, found {len(dataset)}."
    )

    print(
        f"[PASS] {split_name} count = "
        f"{expected_count}"
    )

    class_counts = Counter()

    for sample in dataset.samples:

        metadata = sample["metadata"]

        class_id = metadata["class_id"]

        class_name = CLASS_NAMES[class_id]

        class_counts[class_name] += 1

    print()
    print("Class distribution:")

    for class_name in CLASS_NAMES:

        count = class_counts[class_name]

        print(
            f"  {class_name:<12} {count}"
        )

    if split_name == "train":

        for class_name in CLASS_NAMES:

            assert (
                class_counts[class_name]
                == EXPECTED_TRAIN_PER_CLASS
            ), (
                f"Training class imbalance: "
                f"{class_name} has "
                f"{class_counts[class_name]} "
                f"instead of "
                f"{EXPECTED_TRAIN_PER_CLASS}."
            )

        print()
        print(
            "[PASS] Training split has "
            "250 images per class."
        )

    if split_name == "val":

        for class_name in CLASS_NAMES:

            assert (
                class_counts[class_name]
                == EXPECTED_VAL_PER_CLASS
            ), (
                f"Validation class imbalance: "
                f"{class_name} has "
                f"{class_counts[class_name]} "
                f"instead of "
                f"{EXPECTED_VAL_PER_CLASS}."
            )

        print()
        print(
            "[PASS] Validation split has "
            "150 images per class."
        )

    return dataset


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

    imageclasslist_path = (
        dataset_root
        / "imageclasslist.txt"
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

    print(
        "Split metadata:",
        imageclasslist_path,
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

    if not imageclasslist_path.exists():
        raise FileNotFoundError(
            "Missing official ExDark split file: "
            f"{imageclasslist_path}"
        )

    train_dataset = inspect_split(
        split_name="train",
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        imageclasslist_path=imageclasslist_path,
    )

    val_dataset = inspect_split(
        split_name="val",
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        imageclasslist_path=imageclasslist_path,
    )

    test_dataset = inspect_split(
        split_name="test",
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        imageclasslist_path=imageclasslist_path,
    )

    train_names = {
        sample["image_path"].name.lower()
        for sample in train_dataset.samples
    }

    val_names = {
        sample["image_path"].name.lower()
        for sample in val_dataset.samples
    }

    test_names = {
        sample["image_path"].name.lower()
        for sample in test_dataset.samples
    }

    print()
    print("=" * 70)
    print("SPLIT OVERLAP CHECK")
    print("=" * 70)

    train_val = train_names & val_names
    train_test = train_names & test_names
    val_test = val_names & test_names

    print(
        "Train ∩ Val:",
        len(train_val),
    )

    print(
        "Train ∩ Test:",
        len(train_test),
    )

    print(
        "Val ∩ Test:",
        len(val_test),
    )

    assert len(train_val) == 0
    assert len(train_test) == 0
    assert len(val_test) == 0

    print()
    print(
        "[PASS] No image appears in more than "
        "one official split."
    )

    all_split_names = (
        train_names
        | val_names
        | test_names
    )

    print()
    print("=" * 70)
    print("TOTAL COVERAGE CHECK")
    print("=" * 70)

    print(
        "Train:",
        len(train_names),
    )

    print(
        "Validation:",
        len(val_names),
    )

    print(
        "Test:",
        len(test_names),
    )

    print(
        "Combined:",
        len(all_split_names),
    )

    assert len(all_split_names) == 7363

    print()
    print(
        "[PASS] Official splits cover all "
        "7,363 ExDark images."
    )

    print()
    print("=" * 70)
    print("EXDARK OFFICIAL SPLIT TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()