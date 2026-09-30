import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """
    CALD-Net v0.1 detection loss.

    Total loss:

        L_total =
            L_box
            + L_objectness
            + L_classification
    """

    def __init__(
        self,
        num_classes: int,
        positive_objectness_weight: float = 5.0,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.positive_objectness_weight = (
            positive_objectness_weight
        )

        self.box_loss = nn.SmoothL1Loss(
            reduction="none"
        )

        self.objectness_loss = (
            nn.BCEWithLogitsLoss(
                reduction="none"
            )
        )

        self.classification_loss = (
            nn.BCEWithLogitsLoss(
                reduction="none"
            )
        )

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ):
        total_box_loss = torch.tensor(
            0.0,
            device=next(
                iter(predictions.values())
            ).device,
        )

        total_objectness_loss = torch.tensor(
            0.0,
            device=next(
                iter(predictions.values())
            ).device,
        )

        total_classification_loss = torch.tensor(
            0.0,
            device=next(
                iter(predictions.values())
            ).device,
        )

        for level in [
            "p3",
            "p4",
            "p5",
        ]:

            pred = predictions[level]
            target = targets[level]

            # -------------------------------------------------
            # Positive mask
            # -------------------------------------------------

            positive = target[:, 4:5]

            num_positive = (
                positive.sum().clamp(min=1.0)
            )

            # -------------------------------------------------
            # Box loss
            # -------------------------------------------------

            pred_box = pred[:, 0:4]
            target_box = target[:, 0:4]

            box_error = self.box_loss(
                pred_box,
                target_box,
            )

            box_error = (
                box_error * positive
            )

            total_box_loss += (
                box_error.sum()
                / num_positive
            )

            # -------------------------------------------------
            # Objectness loss
            # -------------------------------------------------

            pred_objectness = pred[:, 4:5]
            target_objectness = target[:, 4:5]

            objectness_error = (
                self.objectness_loss(
                    pred_objectness,
                    target_objectness,
                )
            )

            positive_weight = (
                self.positive_objectness_weight
            )

            negative_weight = 1.0

            weights = torch.where(
                target_objectness == 1,
                torch.full_like(
                    target_objectness,
                    positive_weight,
                ),
                torch.full_like(
                    target_objectness,
                    negative_weight,
                ),
            )

            objectness_error = (
                objectness_error * weights
            )

            total_objectness_loss += (
                objectness_error.mean()
            )

            # -------------------------------------------------
            # Classification loss
            # -------------------------------------------------

            pred_classes = pred[:, 5:]
            target_classes = target[:, 5:]

            class_error = (
                self.classification_loss(
                    pred_classes,
                    target_classes,
                )
            )

            class_error = (
                class_error * positive
            )

            total_classification_loss += (
                class_error.sum()
                / num_positive
            )

        total_loss = (
            total_box_loss
            + total_objectness_loss
            + total_classification_loss
        )

        return {
            "total": total_loss,
            "box": total_box_loss,
            "objectness": total_objectness_loss,
            "classification": total_classification_loss,
        }