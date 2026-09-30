import torch


def box_xywh_to_xyxy(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Convert bounding boxes from:

        x, y, width, height

    to:

        x1, y1, x2, y2
    """

    x, y, w, h = boxes.unbind(dim=-1)

    x1 = x - w / 2
    y1 = y - h / 2

    x2 = x + w / 2
    y2 = y + h / 2

    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=-1,
    )


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate pairwise IoU.

    boxes1:
        [N, 4]

    boxes2:
        [M, 4]

    Returns:
        [N, M]
    """

    top_left = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    bottom_right = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    intersection_wh = (
        bottom_right - top_left
    ).clamp(min=0)

    intersection_area = (
        intersection_wh[..., 0]
        * intersection_wh[..., 1]
    )

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        *
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        *
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union_area = (
        area1[:, None]
        + area2[None, :]
        - intersection_area
    )

    return (
        intersection_area
        / union_area.clamp(min=1e-7)
    )