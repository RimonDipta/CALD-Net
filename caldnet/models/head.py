import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Multi-scale detection head.

    Each prediction location contains:

        4 box values
        1 objectness value
        N class values
    """

    def __init__(self, num_classes: int):
        super().__init__()

        output_channels = 5 + num_classes

        self.head_p3 = nn.Conv2d(
            128,
            output_channels,
            kernel_size=1,
        )

        self.head_p4 = nn.Conv2d(
            128,
            output_channels,
            kernel_size=1,
        )

        self.head_p5 = nn.Conv2d(
            128,
            output_channels,
            kernel_size=1,
        )

    def forward(self, p3, p4, p5):

        pred_p3 = self.head_p3(p3)
        pred_p4 = self.head_p4(p4)
        pred_p5 = self.head_p5(p5)

        return {
            "p3": pred_p3,
            "p4": pred_p4,
            "p5": pred_p5,
        }