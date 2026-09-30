import torch

from caldnet.models.inference import (
    decode_predictions,
    filter_predictions,
)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    num_classes = 3

    # ---------------------------------------------------------
    # Create one fake P3 prediction map
    # ---------------------------------------------------------
    #
    # P3:
    #
    # 80 × 80 = 6,400 locations
    #
    # Channels:
    #
    # 4 box values
    # 1 objectness value
    # 3 class values
    # ---------------------------------------------------------

    prediction = torch.zeros(
        1,
        5 + num_classes,
        80,
        80,
        device=device,
    )

    # ---------------------------------------------------------
    # Make every location initially represent background.
    #
    # A strongly negative logit produces a probability
    # close to zero after sigmoid().
    # ---------------------------------------------------------

    prediction[
        0,
        4,
        :,
        :,
    ] = -10.0

    prediction[
        0,
        5:,
        :,
        :,
    ] = -10.0

    # ---------------------------------------------------------
    # Create ONE strong prediction.
    #
    # Grid location:
    #
    # x = 16
    # y = 24
    # ---------------------------------------------------------

    prediction[
        0,
        0,
        24,
        16,
    ] = 0.0

    prediction[
        0,
        1,
        24,
        16,
    ] = 0.0

    prediction[
        0,
        2,
        24,
        16,
    ] = 0.0

    prediction[
        0,
        3,
        24,
        16,
    ] = 0.0

    # ---------------------------------------------------------
    # Strong objectness
    # ---------------------------------------------------------

    prediction[
        0,
        4,
        24,
        16,
    ] = 4.0

    # ---------------------------------------------------------
    # Class 1 is strongest
    # ---------------------------------------------------------

    prediction[
        0,
        5,
        24,
        16,
    ] = -3.0

    prediction[
        0,
        6,
        24,
        16,
    ] = 4.0

    prediction[
        0,
        7,
        24,
        16,
    ] = -3.0

    # ---------------------------------------------------------
    # Decode
    # ---------------------------------------------------------

    decoded = decode_predictions(
        prediction,
        stride=8,
        num_classes=num_classes,
    )

    # ---------------------------------------------------------
    # Filter
    # ---------------------------------------------------------

    boxes, scores, class_ids = (
        filter_predictions(
            decoded,
            confidence_threshold=0.25,
        )
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\nNumber of detections:")

    print(
        len(boxes)
    )

    print("\nBoxes:")

    print(boxes)

    print("\nScores:")

    print(scores)

    print("\nClass IDs:")

    print(class_ids)


if __name__ == "__main__":
    main()