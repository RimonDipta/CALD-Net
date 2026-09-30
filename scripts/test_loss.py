import torch

from caldnet.models.detector import CALDNet
from caldnet.models.assigner import TargetAssigner
from caldnet.models.targets import TargetBuilder
from caldnet.models.loss import DetectionLoss


def main():

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    num_classes = 3

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = CALDNet(
        num_classes=num_classes
    ).to(device)

    model.train()

    # ---------------------------------------------------------
    # Fake input
    # ---------------------------------------------------------
    #
    # We are deliberately NOT loading images yet.
    #
    # This random tensor represents a batch containing
    # one 640x640 RGB image.
    # ---------------------------------------------------------

    x = torch.randn(
        1,
        3,
        640,
        640,
        device=device,
    )

    # ---------------------------------------------------------
    # Fake ground-truth objects
    # ---------------------------------------------------------
    #
    # Format:
    #
    # [x_center, y_center, width, height, class_id]
    #
    # Coordinates are normalized to [0, 1].
    # ---------------------------------------------------------

    targets = torch.tensor(
        [
            [
                0.20,
                0.30,
                0.10,
                0.10,
                0,
            ],
            [
                0.60,
                0.50,
                0.20,
                0.20,
                1,
            ],
            [
                0.75,
                0.70,
                0.50,
                0.40,
                2,
            ],
        ],
        device=device,
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    predictions = model(x)

    print("\nPredictions:")

    for level, prediction in predictions.items():

        print(
            f"{level.upper()}:",
            prediction.shape,
        )

    # ---------------------------------------------------------
    # Assign targets to feature levels
    # ---------------------------------------------------------

    assigner = TargetAssigner()

    assignments = assigner.assign(
        targets,
        {
            "p3": 80,
            "p4": 40,
            "p5": 20,
        },
    )

    print("\nAssignments:")

    for level, items in assignments.items():

        print(
            f"{level.upper()}:",
            len(items),
            "object(s)",
        )

        for item in items:

            print(
                "  grid:",
                (
                    item["grid_x"],
                    item["grid_y"],
                ),
                "class:",
                item["class_id"],
            )

    # ---------------------------------------------------------
    # Build dense target tensors
    # ---------------------------------------------------------

    target_builder = TargetBuilder(
        num_classes=num_classes
    )

    target_tensors = target_builder.build(
        assignments,
        device,
    )

    # ---------------------------------------------------------
    # Calculate loss
    # ---------------------------------------------------------

    criterion = DetectionLoss(
        num_classes=num_classes
    )

    losses = criterion(
        predictions,
        target_tensors,
    )

    print("\nLosses:")

    print(
        "Box:",
        losses["box"].item(),
    )

    print(
        "Objectness:",
        losses["objectness"].item(),
    )

    print(
        "Classification:",
        losses["classification"].item(),
    )

    print(
        "Total:",
        losses["total"].item(),
    )

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------

    model.zero_grad()

    losses["total"].backward()

    print("\nGradient check:")

    # Check a few important layers.

    layers = {
        "backbone.stage1.conv":
            model.backbone.stage1.block[0].weight,

        "backbone.stage3.conv":
            model.backbone.stage3.block[0].weight,

        "neck.p3_refine":
            model.neck.p3_refine.weight,

        "head.p3":
            model.head.head_p3.weight,
    }

    for name, parameter in layers.items():

        if parameter.grad is None:

            print(
                f"{name}: NO GRADIENT"
            )

        else:

            gradient_mean = (
                parameter.grad
                .abs()
                .mean()
                .item()
            )

            gradient_max = (
                parameter.grad
                .abs()
                .max()
                .item()
            )

            print(
                f"{name}:",
                f"mean={gradient_mean:.8f}",
                f"max={gradient_max:.8f}",
            )


if __name__ == "__main__":
    main()