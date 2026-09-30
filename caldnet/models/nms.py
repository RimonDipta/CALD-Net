import torch

from .box_ops import box_iou


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Perform class-agnostic Non-Maximum Suppression.

    Parameters
    ----------
    boxes:
        Tensor with shape [N, 4].

        Format:

            [x1, y1, x2, y2]

    scores:
        Tensor with shape [N].

    iou_threshold:
        Boxes with IoU greater than this value
        are suppressed.

    Returns
    -------
    keep:
        Tensor containing the indices of boxes
        that survived NMS.
    """

    # ---------------------------------------------------------
    # Handle empty input
    # ---------------------------------------------------------

    if boxes.numel() == 0:

        return torch.empty(
            0,
            dtype=torch.long,
            device=boxes.device,
        )

    # ---------------------------------------------------------
    # Sort boxes by confidence
    # ---------------------------------------------------------

    order = torch.argsort(
        scores,
        descending=True,
    )

    keep = []

    # ---------------------------------------------------------
    # Process boxes
    # ---------------------------------------------------------

    while order.numel() > 0:

        # Highest-confidence box
        current = order[0]

        keep.append(current)

        # Only one box remains
        if order.numel() == 1:
            break

        # -----------------------------------------------------
        # Remaining boxes
        # -----------------------------------------------------

        remaining = order[1:]

        current_box = boxes[
            current
        ].unsqueeze(0)

        remaining_boxes = boxes[
            remaining
        ]

        # -----------------------------------------------------
        # IoU between current box and remaining boxes
        # -----------------------------------------------------

        ious = box_iou(
            current_box,
            remaining_boxes,
        ).squeeze(0)

        # -----------------------------------------------------
        # Keep boxes whose overlap is small enough
        # -----------------------------------------------------

        keep_mask = (
            ious <= iou_threshold
        )

        order = remaining[
            keep_mask
        ]

    return torch.stack(
        keep
    )