import torch

from caldnet.models.decode import (
    decode_boxes,
)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Create one fake prediction
    # ---------------------------------------------------------
    #
    # Shape:
    #
    # [batch, channels, height, width]
    #
    # Here:
    #
    # batch = 1
    # channels = 4
    # grid = 80x80
    # ---------------------------------------------------------

    prediction = torch.zeros(
        1,
        4,
        80,
        80,
        device=device,
    )

    # ---------------------------------------------------------
    # Put one prediction at grid location:
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

    # exp(0) = 1
    #
    # Therefore width and height will be
    # one feature-map cell = 8 pixels.
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
    # Decode
    # ---------------------------------------------------------

    boxes = decode_boxes(
        prediction,
        stride=8,
    )

    # ---------------------------------------------------------
    # Inspect the selected location
    # ---------------------------------------------------------

    box = boxes[
        0,
        24,
        16,
    ]

    print("\nDecoded box:")

    print(box)

    print(
        "\nExpected center approximately:"
    )

    print(
        "x =",
        (16 + 0.5) * 8,
    )

    print(
        "y =",
        (24 + 0.5) * 8,
    )

    print(
        "\nExpected width:",
        8,
    )

    print(
        "Expected height:",
        8,
    )


if __name__ == "__main__":
    main()