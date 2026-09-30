import torch

from caldnet.models.detector import CALDNet

from caldnet.models.detector_inference import (
    run_inference,
)


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

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = CALDNet(
        num_classes=num_classes
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Random input
    # ---------------------------------------------------------

    x = torch.randn(
        1,
        3,
        640,
        640,
        device=device,
    )

    # ---------------------------------------------------------
    # Model prediction
    # ---------------------------------------------------------

    with torch.no_grad():

        predictions = model(x)

    # ---------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------

    detections = run_inference(
        predictions,
        num_classes=num_classes,
        confidence_threshold=0.50,
        iou_threshold=0.50,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    result = detections[0]

    print("\nFinal detections:")

    print(
        "Number:",
        len(result["boxes"]),
    )

    print(
        "\nBoxes:"
    )

    print(
        result["boxes"]
    )

    print(
        "\nScores:"
    )

    print(
        result["scores"]
    )

    print(
        "\nClass IDs:"
    )

    print(
        result["class_ids"]
    )


if __name__ == "__main__":
    main()