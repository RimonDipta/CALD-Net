import torch
from torch.utils.data import DataLoader

from caldnet.data.collate import detection_collate_fn
from caldnet.data.exdark import ExDarkDataset
from caldnet.models.assigner import TargetAssigner
from caldnet.models.detector import CALDNet
from caldnet.models.loss import DetectionLoss
from caldnet.models.targets import TargetBuilder


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    num_classes = 12
    batch_size = 4

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ExDarkDataset(
        image_dir="data/exdark/images",
        annotation_dir="data/exdark/annotations",
        image_size=640,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    images, targets = next(
        iter(dataloader)
    )

    print()
    print(
        "Input images:",
        tuple(images.shape),
    )

    print(
        "Number of target lists:",
        len(targets),
    )

    for index, target in enumerate(targets):

        print(
            f"Image {index} objects:",
            target.shape[0],
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CALDNet(
        num_classes=num_classes
    ).to(device)

    model.eval()

    images = images.to(device)

    with torch.no_grad():

        predictions = model(
            images
        )

    print()
    print("Model outputs:")

    for level in [
        "p3",
        "p4",
        "p5",
    ]:

        print(
            f"  {level}:",
            tuple(
                predictions[level].shape
            ),
        )

    # --------------------------------------------------------
    # Target assignment
    # --------------------------------------------------------

    assigner = TargetAssigner()

    grid_sizes = {
        "p3": 80,
        "p4": 40,
        "p5": 20,
    }

    batch_assignments = {
        "p3": [],
        "p4": [],
        "p5": [],
    }

    for target in targets:

        assignments = assigner.assign(
            targets=target,
            grid_sizes=grid_sizes,
        )

        for level in [
            "p3",
            "p4",
            "p5",
        ]:

            batch_assignments[
                level
            ].append(
                assignments[level]
            )

    print()
    print(
        "Assignments:"
    )

    for level in [
        "p3",
        "p4",
        "p5",
    ]:

        counts = [
            len(image_assignments)
            for image_assignments
            in batch_assignments[level]
        ]

        print(
            f"  {level}:",
            counts,
        )

    # --------------------------------------------------------
    # Target construction
    # --------------------------------------------------------

    target_builder = TargetBuilder(
        num_classes=num_classes
    )

    dense_targets = (
        target_builder.build_batch(
            batch_assignments=batch_assignments,
            device=device,
        )
    )

    print()
    print(
        "Dense targets:"
    )

    for level in [
        "p3",
        "p4",
        "p5",
    ]:

        print(
            f"  {level}:",
            tuple(
                dense_targets[level].shape
            ),
        )

    # --------------------------------------------------------
    # Validate target/prediction shapes
    # --------------------------------------------------------

    for level in [
        "p3",
        "p4",
        "p5",
    ]:

        assert (
            predictions[level].shape
            == dense_targets[level].shape
        ), (
            f"Shape mismatch at {level}: "
            f"prediction="
            f"{predictions[level].shape}, "
            f"target="
            f"{dense_targets[level].shape}"
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_function = DetectionLoss(
        num_classes=num_classes
    ).to(device)

    losses = loss_function(
        predictions,
        dense_targets,
    )

    print()
    print(
        "Loss:"
    )

    print(
        "  Total:",
        float(losses["total"].item()),
    )

    print(
        "  Box:",
        float(losses["box"].item()),
    )

    print(
        "  Objectness:",
        float(losses["objectness"].item()),
    )

    print(
        "  Classification:",
        float(
            losses["classification"].item()
        ),
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert torch.isfinite(
        losses["total"]
    )

    assert torch.isfinite(
        losses["box"]
    )

    assert torch.isfinite(
        losses["objectness"]
    )

    assert torch.isfinite(
        losses["classification"]
    )

    print()
    print(
        "[PASS] Real ExDark batch successfully "
        "passed through CALD-Net."
    )

    print()
    print(
        "Pipeline:"
    )

    print(
        "ExDark -> DataLoader -> "
        "CALD-Net -> Assignment -> "
        "Targets -> Loss"
    )


if __name__ == "__main__":
    main()