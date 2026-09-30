import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from caldnet.data.synthetic import (
    SyntheticDetectionDataset,
)

from caldnet.data.collate import (
    detection_collate_fn,
)

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
    image_size = 640
    batch_size = 2
    epochs = 3

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = SyntheticDetectionDataset(
        num_samples=10,
        num_classes=num_classes,
        image_size=image_size,
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
    )

    # ---------------------------------------------------------
    # Model
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

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # ---------------------------------------------------------
    # Target utilities
    # ---------------------------------------------------------

    assigner = TargetAssigner()

    target_builder = TargetBuilder(
        num_classes=num_classes
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print("\nTraining:\n")

    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_loss = 0.0

        print(
            f"Epoch {epoch}/{epochs}"
        )

        for batch_index, (
            images,
            targets,
        ) in enumerate(loader, start=1):

            # -------------------------------------------------
            # Move images to GPU
            # -------------------------------------------------

            images = images.to(device)

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------

            predictions = model(images)

            # -------------------------------------------------
            # Calculate loss for each image
            # -------------------------------------------------
            #
            # Our current TargetBuilder only handles one image
            # at a time.
            #
            # Therefore we calculate each sample's target
            # tensors separately.
            # -------------------------------------------------

            batch_loss = torch.tensor(
                0.0,
                device=device,
            )

            for image_index, image_targets in enumerate(
                targets
            ):

                # ---------------------------------------------
                # Assign this image's objects
                # ---------------------------------------------

                assignments = assigner.assign(
                    image_targets.to(device),
                    {
                        "p3": 80,
                        "p4": 40,
                        "p5": 20,
                    },
                )

                # ---------------------------------------------
                # Build target tensors
                # ---------------------------------------------

                target_tensors = target_builder.build(
                    assignments,
                    device,
                )

                # ---------------------------------------------
                # Select predictions for this image
                # ---------------------------------------------

                image_predictions = {
                    "p3": predictions["p3"][
                        image_index:image_index + 1
                    ],
                    "p4": predictions["p4"][
                        image_index:image_index + 1
                    ],
                    "p5": predictions["p5"][
                        image_index:image_index + 1
                    ],
                }

                # ---------------------------------------------
                # Calculate image loss
                # ---------------------------------------------

                losses = criterion(
                    image_predictions,
                    target_tensors,
                )

                batch_loss = (
                    batch_loss
                    + losses["total"]
                )

            # -------------------------------------------------
            # Average batch loss
            # -------------------------------------------------

            batch_loss = (
                batch_loss
                / len(targets)
            )

            # -------------------------------------------------
            # Clear old gradients
            # -------------------------------------------------

            optimizer.zero_grad()

            # -------------------------------------------------
            # Backward pass
            # -------------------------------------------------

            batch_loss.backward()

            # -------------------------------------------------
            # Update weights
            # -------------------------------------------------

            optimizer.step()

            # -------------------------------------------------
            # Record loss
            # -------------------------------------------------

            epoch_loss += (
                batch_loss.item()
            )

            print(
                f"  Batch {batch_index}: "
                f"loss={batch_loss.item():.4f}"
            )

        # -----------------------------------------------------
        # Average epoch loss
        # -----------------------------------------------------

        average_loss = (
            epoch_loss
            / len(loader)
        )

        print(
            f"Epoch {epoch} average loss: "
            f"{average_loss:.4f}\n"
        )


if __name__ == "__main__":
    main()