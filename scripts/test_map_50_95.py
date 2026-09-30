import torch

from caldnet.evaluation.map import (
    evaluate_map_50_95,
)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Ground truth
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

    results = evaluate_map_50_95(
        predicted_boxes=predicted_boxes,
        predicted_scores=predicted_scores,
        predicted_classes=predicted_classes,
        target_boxes=target_boxes,
        target_classes=target_classes,
        num_classes=3,
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\nResults")
    print("-------")

    print(
        "mAP@50:",
        results["map_50"],
    )

    print(
        "mAP@50:95:",
        results["map_50_95"],
    )

    print("\nAP at each IoU threshold:")

    for threshold, map_value in results[
        "map_per_iou"
    ].items():

        print(
            f"IoU {threshold:.2f}: "
            f"{map_value:.4f}"
        )


if __name__ == "__main__":
    main()