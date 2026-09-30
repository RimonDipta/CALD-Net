import torch

from .metrics import evaluate_map


def evaluate_map_50_95(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    num_classes: int,
):
    """
    Calculate COCO-style mAP@50:95.

    IoU thresholds:

        0.50
        0.55
        0.60
        0.65
        0.70
        0.75
        0.80
        0.85
        0.90
        0.95

    Returns
    -------
    dict
        {
            "map_50_95": float,
            "map_50": float,
            "map_per_iou": dict,
        }
    """

    iou_thresholds = torch.arange(
        0.50,
        0.951,
        0.05,
        dtype=torch.float32,
    )

    map_per_iou = {}

    for iou_threshold in iou_thresholds:

        threshold = float(
            iou_threshold.item()
        )

        results = evaluate_map(
            predicted_boxes=predicted_boxes,
            predicted_scores=predicted_scores,
            predicted_classes=predicted_classes,
            target_boxes=target_boxes,
            target_classes=target_classes,
            num_classes=num_classes,
            iou_threshold=threshold,
        )

        map_per_iou[threshold] = results[
            "map"
        ]

    # ---------------------------------------------------------
    # Average AP across all IoU thresholds.
    # ---------------------------------------------------------

    map_50_95 = sum(
        map_per_iou.values()
    ) / len(map_per_iou)

    map_50 = map_per_iou[0.50]

    return {
        "map_50_95": map_50_95,
        "map_50": map_50,
        "map_per_iou": map_per_iou,
    }