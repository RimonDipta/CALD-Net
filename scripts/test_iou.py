import torch

from caldnet.models.box_ops import box_iou


def main():

    box_a = torch.tensor(
        [
            [
                10.0,
                10.0,
                50.0,
                50.0,
            ]
        ]
    )

    box_b = torch.tensor(
        [
            [
                20.0,
                20.0,
                60.0,
                60.0,
            ]
        ]
    )

    iou = box_iou(
        box_a,
        box_b,
    )

    print("Box A:", box_a)
    print("Box B:", box_b)
    print("IoU:", iou)


if __name__ == "__main__":
    main()