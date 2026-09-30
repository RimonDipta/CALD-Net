import torch
import torch.optim as optim

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

    num_steps = 20

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = CALDNet(
        num_classes=num_classes
    ).to(device)

    model.train()

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    criterion = DetectionLoss(
        num_classes=num_classes
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    #
    # Adam changes the model parameters using
    # the gradients produced by backward().
    # ---------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # ---------------------------------------------------------
    # Target tools
    # ---------------------------------------------------------

    assigner = TargetAssigner()

    target_builder = TargetBuilder(
        num_classes=num_classes
    )

    # ---------------------------------------------------------
    # Fixed synthetic input
    # ---------------------------------------------------------
    #
    # We keep the input and target fixed.
    #
    # This makes it easier to observe whether the
    # model can learn the same artificial example.
    # ---------------------------------------------------------

    x = torch.randn(
        1,
        3,
        640,
        640,
        device=device,
    )

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
    # Build targets
    # ---------------------------------------------------------

    assignments = assigner.assign(
        targets,
        {
            "p3": 80,
            "p4": 40,
            "p5": 20,
        },
    )

    target_tensors = target_builder.build(
        assignments,
        device,
    )

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    print("\nTraining:\n")

    for step in range(
        1,
        num_steps + 1,
    ):

        # -----------------------------------------------------
        # Clear old gradients
        # -----------------------------------------------------

        optimizer.zero_grad()

        # -----------------------------------------------------
        # Forward
        # -----------------------------------------------------

        predictions = model(x)

        # -----------------------------------------------------
        # Loss
        # -----------------------------------------------------

        losses = criterion(
            predictions,
            target_tensors,
        )

        total_loss = losses["total"]

        # -----------------------------------------------------
        # Backward
        # -----------------------------------------------------

        total_loss.backward()

        # -----------------------------------------------------
        # Update model parameters
        # -----------------------------------------------------

        optimizer.step()

        # -----------------------------------------------------
        # Print progress
        # -----------------------------------------------------

        print(
            f"Step {step:02d} | "
            f"Total: {losses['total'].item():.4f} | "
            f"Box: {losses['box'].item():.4f} | "
            f"Objectness: {losses['objectness'].item():.4f} | "
            f"Class: {losses['classification'].item():.4f}"
        )


if __name__ == "__main__":
    main()