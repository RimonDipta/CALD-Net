import torch

from caldnet.evaluation.detection import (
    evaluate_detections,
)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Ground-truth objects
    # ---------------------------------------------------------

    target_boxes = torch.tensor(
        [
            [
                100.0,
                100.0,
                200.0,
                200.0,
            ],
            [
                400.0,
                400.0,
                500.0,
                500.0,
            ],
            [
                600.0,
                100.0,
                640.0,
                200.0,
            ],
        ],
        device=device,
    )

    target_classes = torch.tensor(
        [
            0,
            1,
            2,
        ],
        device=device,
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    #
    # Prediction 0:
    # Correct class and good overlap with target 0.
    #
    # Prediction 1:
    # Correct class and good overlap with target 1.
    #
    # Prediction 2:
    # Wrong class for target 2.
    #
    # Prediction 3:
    # No corresponding ground-truth object.
    # ---------------------------------------------------------

    predicted_boxes = torch.tensor(
        [
            [
                110.0,
                105.0,
                195.0,
                200.0,
            ],
            [
                405.0,
                405.0,
                495.0,
                495.0,
            ],
            [
                600.0,
                100.0,
                640.0,
                200.0,
            ],
            [
                250.0,
                250.0,
                300.0,
                300.0,
            ],
        ],
        device=device,
    )

    predicted_scores = torch.tensor(
        [
            0.95,
            0.90,
            0.85,
            0.40,
        ],
        device=device,
    )

    predicted_classes = torch.tensor(
        [
            0,
            1,
            1,
            0,
        ],
        device=device,
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    result = evaluate_detections(
        predicted_boxes,
        predicted_scores,
        predicted_classes,
        target_boxes,
        target_classes,
        iou_threshold=0.50,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\nEvaluation:")

    print(
        "True positives:",
        result["true_positives"],
    )

    print(
        "False positives:",
        result["false_positives"],
    )

    print(
        "False negatives:",
        result["false_negatives"],
    )

    print(
        "Precision:",
        result["precision"],
    )

    print(
        "Recall:",
        result["recall"],
    )


if __name__ == "__main__":
    main()