import torch

from .decode import decode_boxes


def decode_predictions(
    prediction: torch.Tensor,
    stride: int,
    num_classes: int,
):
    """
    Convert one feature-level prediction tensor into
    candidate detections.

    Input:

        prediction:
            [B, 5 + num_classes, H, W]

    Output:

        boxes:
            [B, H, W, 4]

        objectness:
            [B, H, W]

        class_probabilities:
            [B, H, W, num_classes]

        confidence:
            [B, H, W, num_classes]
    """

    # ---------------------------------------------------------
    # Decode bounding boxes
    # ---------------------------------------------------------

    boxes = decode_boxes(
        prediction[:, 0:4],
        stride=stride,
    )

    # ---------------------------------------------------------
    # Objectness
    # ---------------------------------------------------------
    #
    # Original shape:
    #
    # [B, H, W]
    # ---------------------------------------------------------

    objectness = torch.sigmoid(
        prediction[:, 4]
    )

    # ---------------------------------------------------------
    # Class probabilities
    # ---------------------------------------------------------
    #
    # Original shape:
    #
    # [B, num_classes, H, W]
    #
    # We permute it to:
    #
    # [B, H, W, num_classes]
    #
    # so the class dimension is last.
    # ---------------------------------------------------------

    class_logits = prediction[
        :,
        5:5 + num_classes,
    ]

    class_probabilities = torch.sigmoid(
        class_logits
    )

    class_probabilities = (
        class_probabilities.permute(
            0,
            2,
            3,
            1,
        )
    )

    # ---------------------------------------------------------
    # Combined confidence
    # ---------------------------------------------------------
    #
    # Objectness:
    #
    # [B, H, W]
    #
    # becomes:
    #
    # [B, H, W, 1]
    #
    # Then it broadcasts across all classes.
    # ---------------------------------------------------------

    confidence = (
        objectness.unsqueeze(-1)
        * class_probabilities
    )

    return {
        "boxes": boxes,
        "objectness": objectness,
        "class_probabilities": class_probabilities,
        "confidence": confidence,
    }


def filter_predictions(
    decoded: dict,
    confidence_threshold: float = 0.25,
):
    """
    Filter candidate detections using confidence.

    Returns:

        boxes:
            [N, 4]

        scores:
            [N]

        class_ids:
            [N]
    """

    boxes = decoded["boxes"]
    confidence = decoded["confidence"]

    # ---------------------------------------------------------
    # This educational implementation handles one image.
    # ---------------------------------------------------------

    boxes = boxes[0]
    confidence = confidence[0]

    # ---------------------------------------------------------
    # Find the highest-confidence class for every location
    # ---------------------------------------------------------

    scores, class_ids = confidence.max(
        dim=-1
    )

    # ---------------------------------------------------------
    # Keep predictions above threshold
    # ---------------------------------------------------------

    keep = scores >= confidence_threshold

    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    return (
        boxes,
        scores,
        class_ids,
    )