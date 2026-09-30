import torch

from caldnet.models.nms import nms


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Three boxes
    # ---------------------------------------------------------
    #
    # Box A and B overlap heavily.
    #
    # Box C is far away.
    # ---------------------------------------------------------

    boxes = torch.tensor(
        [
            [
                100.0,
                100.0,
                200.0,
                200.0,
            ],
            [
                110.0,
                110.0,
                210.0,
                210.0,
            ],
            [
                400.0,
                400.0,
                500.0,
                500.0,
            ],
        ],
        device=device,
    )

    # ---------------------------------------------------------
    # Confidence scores
    # ---------------------------------------------------------

    scores = torch.tensor(
        [
            0.95,
            0.80,
            0.70,
        ],
        device=device,
    )

    print("\nInput boxes:")

    print(boxes)

    print("\nScores:")

    print(scores)

    # ---------------------------------------------------------
    # Calculate IoU between first two boxes
    # ---------------------------------------------------------

    from caldnet.models.box_ops import box_iou

    iou = box_iou(
        boxes[0:1],
        boxes[1:2],
    )

    print("\nIoU between Box 0 and Box 1:")

    print(iou)

    # ---------------------------------------------------------
    # NMS
    # ---------------------------------------------------------

    keep = nms(
        boxes,
        scores,
        iou_threshold=0.5,
    )

    print("\nKept indices:")

    print(keep)

    print("\nKept boxes:")

    print(boxes[keep])

    print("\nKept scores:")

    print(scores[keep])


if __name__ == "__main__":
    main()