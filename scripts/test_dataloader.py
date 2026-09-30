from torch.utils.data import DataLoader

from caldnet.data.synthetic import (
    SyntheticDetectionDataset,
)

from caldnet.data.collate import (
    detection_collate_fn,
)


def main():

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = SyntheticDetectionDataset(
        num_samples=10,
        num_classes=3,
        image_size=640,
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=detection_collate_fn,
    )

    # ---------------------------------------------------------
    # Get first batch
    # ---------------------------------------------------------

    images, targets = next(
        iter(loader)
    )

    # ---------------------------------------------------------
    # Inspect batch
    # ---------------------------------------------------------

    print("Images:")
    print(
        "Shape:",
        images.shape,
    )

    print(
        "Dtype:",
        images.dtype,
    )

    print("\nTargets:")
    print(
        "Type:",
        type(targets),
    )

    print(
        "Number of target tensors:",
        len(targets),
    )

    # ---------------------------------------------------------
    # Inspect individual targets
    # ---------------------------------------------------------

    for index, target in enumerate(targets):

        print(
            f"\nImage {index}:"
        )

        print(
            "Target shape:",
            target.shape,
        )

        print(
            target
        )


if __name__ == "__main__":
    main()