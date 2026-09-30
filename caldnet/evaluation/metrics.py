import torch

from caldnet.models.box_ops import box_iou


def compute_ap(
    precisions: torch.Tensor,
    recalls: torch.Tensor,
) -> float:
    """
    Compute Average Precision from a precision-recall curve.

    Uses the interpolated precision envelope.

    Parameters
    ----------
    precisions:
        Precision values ordered by increasing recall.

    recalls:
        Recall values ordered by increasing recall.

    Returns
    -------
    float
        Average Precision.
    """

    if precisions.numel() == 0:
        return 0.0

    # ---------------------------------------------------------
    # Add boundary points.
    # ---------------------------------------------------------

    precision_curve = torch.cat(
        [
            torch.tensor(
                [1.0],
                device=precisions.device,
            ),
            precisions,
            torch.tensor(
                [0.0],
                device=precisions.device,
            ),
        ]
    )

    recall_curve = torch.cat(
        [
            torch.tensor(
                [0.0],
                device=recalls.device,
            ),
            recalls,
            torch.tensor(
                [1.0],
                device=recalls.device,
            ),
        ]
    )

    # ---------------------------------------------------------
    # Create the precision envelope.
    #
    # Moving from right to left:
    #
    # precision[i] =
    #     max(precision[i], precision[i+1])
    # ---------------------------------------------------------

    for index in range(
        precision_curve.numel() - 2,
        -1,
        -1,
    ):
        precision_curve[index] = torch.maximum(
            precision_curve[index],
            precision_curve[index + 1],
        )

    # ---------------------------------------------------------
    # Find recall positions where recall changes.
    # ---------------------------------------------------------

    recall_change = (
        recall_curve[1:]
        != recall_curve[:-1]
    )

    recall_indices = torch.nonzero(
        recall_change,
        as_tuple=False,
    ).squeeze(1)

    # ---------------------------------------------------------
    # Integrate precision over recall.
    # ---------------------------------------------------------

    ap = torch.sum(
        (
            recall_curve[
                recall_indices + 1
            ]
            - recall_curve[
                recall_indices
            ]
        )
        * precision_curve[
            recall_indices + 1
        ]
    )

    return float(ap.item())


def evaluate_class(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    class_id: int,
    iou_threshold: float = 0.50,
):
    """
    Evaluate one class across all confidence thresholds.

    Predictions are sorted by confidence and progressively
    considered as the confidence threshold is lowered.

    Returns:
        precision
        recall
        ap
        true_positives
        false_positives
        num_targets
    """

    # ---------------------------------------------------------
    # Select predictions belonging to this class.
    # ---------------------------------------------------------

    prediction_mask = (
        predicted_classes == class_id
    )

    boxes = predicted_boxes[
        prediction_mask
    ]

    scores = predicted_scores[
        prediction_mask
    ]

    # ---------------------------------------------------------
    # Select ground-truth objects belonging to this class.
    # ---------------------------------------------------------

    target_mask = (
        target_classes == class_id
    )

    targets = target_boxes[
        target_mask
    ]

    num_targets = targets.shape[0]

    # ---------------------------------------------------------
    # No predictions for this class.
    # ---------------------------------------------------------

    if boxes.numel() == 0:

        if num_targets == 0:
            return {
                "precision": torch.empty(0),
                "recall": torch.empty(0),
                "ap": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "num_targets": 0,
            }

        return {
            "precision": torch.tensor([0.0]),
            "recall": torch.tensor([0.0]),
            "ap": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "num_targets": num_targets,
        }

    # ---------------------------------------------------------
    # Sort predictions by confidence.
    # ---------------------------------------------------------

    order = torch.argsort(
        scores,
        descending=True,
    )

    boxes = boxes[order]
    scores = scores[order]

    # ---------------------------------------------------------
    # Calculate IoU between every prediction and target.
    # ---------------------------------------------------------

    if num_targets > 0:

        ious = box_iou(
            boxes,
            targets,
        )

    else:

        ious = torch.zeros(
            boxes.shape[0],
            0,
            device=boxes.device,
        )

    matched_targets = torch.zeros(
        num_targets,
        dtype=torch.bool,
        device=boxes.device,
    )

    true_positive_flags = []
    false_positive_flags = []

    # ---------------------------------------------------------
    # Match each prediction.
    # ---------------------------------------------------------

    for prediction_index in range(
        boxes.shape[0]
    ):

        if num_targets == 0:

            true_positive_flags.append(0)
            false_positive_flags.append(1)

            continue

        prediction_ious = ious[
            prediction_index
        ]

        available_targets = (
            ~matched_targets
        )

        candidate_ious = prediction_ious.clone()

        candidate_ious[
            ~available_targets
        ] = -1.0

        best_iou, best_target = (
            candidate_ious.max(
                dim=0
            )
        )

        if best_iou >= iou_threshold:

            true_positive_flags.append(1)
            false_positive_flags.append(0)

            matched_targets[
                best_target
            ] = True

        else:

            true_positive_flags.append(0)
            false_positive_flags.append(1)

    # ---------------------------------------------------------
    # Convert match results to tensors.
    # ---------------------------------------------------------

    true_positive_flags = torch.tensor(
        true_positive_flags,
        dtype=torch.float32,
        device=boxes.device,
    )

    false_positive_flags = torch.tensor(
        false_positive_flags,
        dtype=torch.float32,
        device=boxes.device,
    )

    # ---------------------------------------------------------
    # Cumulative TP / FP.
    #
    # Example:
    #
    # TP = [1, 0, 1]
    #
    # cumulative TP = [1, 1, 2]
    # ---------------------------------------------------------

    cumulative_tp = torch.cumsum(
        true_positive_flags,
        dim=0,
    )

    cumulative_fp = torch.cumsum(
        false_positive_flags,
        dim=0,
    )

    # ---------------------------------------------------------
    # Precision at every confidence threshold.
    # ---------------------------------------------------------

    precision = (
        cumulative_tp
        / (
            cumulative_tp
            + cumulative_fp
        ).clamp(min=1e-7)
    )

    # ---------------------------------------------------------
    # Recall at every confidence threshold.
    # ---------------------------------------------------------

    if num_targets > 0:

        recall = (
            cumulative_tp
            / float(num_targets)
        )

    else:

        recall = torch.zeros_like(
            cumulative_tp
        )

    # ---------------------------------------------------------
    # AP.
    # ---------------------------------------------------------

    ap = compute_ap(
        precision,
        recall,
    )

    return {
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "true_positives": int(
            cumulative_tp[-1].item()
        ),
        "false_positives": int(
            cumulative_fp[-1].item()
        ),
        "num_targets": num_targets,
    }


def evaluate_map(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.50,
):
    """
    Calculate AP for every class and mAP.

    Returns:
        {
            "map": float,
            "class_metrics": {...}
        }
    """

    class_metrics = {}

    valid_aps = []

    for class_id in range(
        num_classes
    ):

        metrics = evaluate_class(
            predicted_boxes=predicted_boxes,
            predicted_scores=predicted_scores,
            predicted_classes=predicted_classes,
            target_boxes=target_boxes,
            target_classes=target_classes,
            class_id=class_id,
            iou_threshold=iou_threshold,
        )

        class_metrics[class_id] = metrics

        # -----------------------------------------------------
        # Only classes with ground-truth objects contribute
        # to mAP.
        # -----------------------------------------------------

        if metrics["num_targets"] > 0:
            valid_aps.append(
                metrics["ap"]
            )

    if len(valid_aps) > 0:

        map_value = sum(valid_aps) / len(
            valid_aps
        )

    else:

        map_value = 0.0

    return {
        "map": map_value,
        "class_metrics": class_metrics,
    }