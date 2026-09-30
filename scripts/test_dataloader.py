import torch
from torch.utils.data import DataLoader

from caldnet.data.exdark import ExDarkDataset
from caldnet.data.collate import detection_collate_fn


def main():

    image_dir = "data/exdark/images"
    annotation_dir = "data/exdark/annotations"

    dataset = ExDarkDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        image_size=640,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    print(
        "Dataset size:",
        len(dataset),
    )

    print(
        "Number of batches:",
        len(dataloader),
    )

    images, targets = next(
        iter(dataloader)
    )

    print()
    print(
        "Images:"
    )

    print(
        "  Shape:",
        tuple(images.shape),
    )

    print(
        "  Dtype:",
        images.dtype,
    )

    print(
        "  Min:",
        float(images.min()),
    )

    print(
        "  Max:",
        float(images.max()),
    )

    print()
    print(
        "Targets:"
    )

    print(
        "  Type:",
        type(targets),
    )

    print(
        "  Number of target tensors:",
        len(targets),
    )

    for index, target in enumerate(
        targets
    ):

        print(
            f"  Image {index}:",
            tuple(target.shape),
        )

        if target.numel() > 0:

            print(
                f"    Classes:",
                sorted(
                    set(
                        target[:, 4]
                        .long()
                        .tolist()
                    )
                ),
            )

    assert images.shape == (
        4,
        3,
        640,
        640,
    )

    assert len(targets) == 4

    for target in targets:

        assert target.ndim == 2
        assert target.shape[1] == 5

        if target.numel() > 0:

            coordinates = target[:, :4]

            assert torch.all(
                coordinates >= 0.0
            )

            assert torch.all(
                coordinates <= 1.0
            )

    print()
    print(
        "[PASS] ExDark DataLoader test."
    )


if __name__ == "__main__":
    main()