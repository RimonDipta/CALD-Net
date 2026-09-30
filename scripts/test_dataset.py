import torch

from caldnet.data.synthetic import (
    SyntheticDetectionDataset,
)


def main():

    # ---------------------------------------------------------
    # Create dataset
    # ---------------------------------------------------------

    dataset = SyntheticDetectionDataset(
        num_samples=5,
        num_classes=3,
        image_size=640,
    )

    print("Dataset size:", len(dataset))

    # ---------------------------------------------------------
    # Get one sample
    # ---------------------------------------------------------

    image, targets = dataset[0]

    print("\nSample:")

    print(
        "Image shape:",
        image.shape,
    )

    print(
        "Image dtype:",
        image.dtype,
    )

    print(
        "Target shape:",
        targets.shape,
    )

    print(
        "Target dtype:",
        targets.dtype,
    )

    print("\nTargets:")

    print(targets)

    # ---------------------------------------------------------
    # Verify target range
    # ---------------------------------------------------------

    coordinates = targets[:, :4]

    print("\nCoordinate range:")

    print(
        "Minimum:",
        coordinates.min().item(),
    )

    print(
        "Maximum:",
        coordinates.max().item(),
    )

    # ---------------------------------------------------------
    # Verify classes
    # ---------------------------------------------------------

    classes = targets[:, 4]

    print("\nClasses:")

    print(classes)

    print(
        "\nClasses are integers:",
        torch.all(
            classes == classes.long()
        ).item(),
    )


if __name__ == "__main__":
    main()