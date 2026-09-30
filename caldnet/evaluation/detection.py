import torch

from caldnet.models.box_ops import box_iou


def evaluate_detections(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Evaluate detections for one image.

    Parameters
    ----------
    predicted_boxes:
        [N, 4]

        Format:

            [x1, y1, x2, y2]

    predicted_scores:
        [N]

        Detection confidence scores.

    predicted_classes:
        [N]

        Predicted class IDs.

    target_boxes:
        [M, 4]

        Ground-truth boxes.

    target_classes:
        [M]

        Ground-truth class IDs.

    iou_threshold:
        Minimum IoU required for a prediction
        to match a ground-truth object.

    Returns
    -------
    result:
        Dictionary containing:

            true_positives
            false_positives
            false_negatives
            precision
            recall
    """

    # ---------------------------------------------------------
    # Empty prediction case
    # ---------------------------------------------------------

    if predicted_boxes.numel() == 0:

        false_negatives = target_boxes.shape[0]

        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": false_negatives,
            "precision": 0.0,
            "recall": (
                0.0
                if false_negatives > 0
                else 1.0
            ),
        }

    # ---------------------------------------------------------
    # Empty target case
    # ---------------------------------------------------------

    if target_boxes.numel() == 0:

        false_positives = predicted_boxes.shape[0]

        return {
            "true_positives": 0,
            "false_positives": false_positives,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 1.0,
        }

    # ---------------------------------------------------------
    # Sort predictions by confidence.
    #
    # Stronger predictions get the opportunity to match first.
    # ---------------------------------------------------------

    order = torch.argsort(
        predicted_scores,
        descending=True,
    )

    predicted_boxes = predicted_boxes[
        order
    ]

    predicted_scores = predicted_scores[
        order
    ]

    predicted_classes = predicted_classes[
        order
    ]

    # ---------------------------------------------------------
    # Calculate all pairwise IoUs
    # ---------------------------------------------------------

    ious = box_iou(
        predicted_boxes,
        target_boxes,
    )

    # ---------------------------------------------------------
    # Keep track of ground-truth objects that
    # have already been matched.
    # ---------------------------------------------------------

    matched_targets = torch.zeros(
        target_boxes.shape[0],
        dtype=torch.bool,
        device=target_boxes.device,
    )

    true_positives = 0
    false_positives = 0

    # ---------------------------------------------------------
    # Match predictions
    # ---------------------------------------------------------

    for prediction_index in range(
        predicted_boxes.shape[0]
    ):

        prediction_class = (
            predicted_classes[
                prediction_index
            ]
        )

        prediction_ious = ious[
            prediction_index
        ]

        # -----------------------------------------------------
        # A prediction can only match a ground-truth object
        # belonging to the same class.
        # -----------------------------------------------------

        class_matches = (
            target_classes
            == prediction_class
        )

        # -----------------------------------------------------
        # Already-matched targets cannot be matched again.
        # -----------------------------------------------------

        available_targets = (
            ~matched_targets
        )

        valid_matches = (
            class_matches
            & available_targets
        )

        # -----------------------------------------------------
        # No valid target of this class remains.
        # -----------------------------------------------------

        if not valid_matches.any():

            false_positives += 1
            continue

        # -----------------------------------------------------
        # Find the highest-IoU valid target.
        # -----------------------------------------------------

        candidate_ious = prediction_ious.clone()

        candidate_ious[
            ~valid_matches
        ] = -1.0

        best_iou, best_target = (
            candidate_ious.max(
                dim=0
            )
        )

        # -----------------------------------------------------
        # Check IoU threshold.
        # -----------------------------------------------------

        if best_iou >= iou_threshold:

            true_positives += 1

            matched_targets[
                best_target
            ] = True

        else:

            false_positives += 1

    # ---------------------------------------------------------
    # False negatives
    # ---------------------------------------------------------

    false_negatives = (
        target_boxes.shape[0]
        - true_positives
    )

    # ---------------------------------------------------------
    # Precision
    # ---------------------------------------------------------

    precision_denominator = (
        true_positives
        + false_positives
    )

    if precision_denominator > 0:

        precision = (
            true_positives
            / precision_denominator
        )

    else:

        precision = 0.0

    # ---------------------------------------------------------
    # Recall
    # ---------------------------------------------------------

    recall_denominator = (
        true_positives
        + false_negatives
    )

    if recall_denominator > 0:

        recall = (
            true_positives
            / recall_denominator
        )

    else:

        recall = 0.0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
    }