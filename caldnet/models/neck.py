import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Lightweight top-down Feature Pyramid Network.
    """

    def __init__(self):
        super().__init__()

        self.p5_reduce = nn.Conv2d(
            512,
            128,
            kernel_size=1,
        )

        self.p4_reduce = nn.Conv2d(
            256,
            128,
            kernel_size=1,
        )

        self.p3_reduce = nn.Conv2d(
            128,
            128,
            kernel_size=1,
        )

        self.p4_refine = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
        )

        self.p3_refine = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
        )

    def forward(self, p3, p4, p5):

        # P5
        p5 = self.p5_reduce(p5)

        # P5 -> P4
        p5_up = F.interpolate(
            p5,
            scale_factor=2,
            mode="nearest",
        )

        p4 = self.p4_reduce(p4)

        p4 = p4 + p5_up
        p4 = self.p4_refine(p4)

        # P4 -> P3
        p4_up = F.interpolate(
            p4,
            scale_factor=2,
            mode="nearest",
        )

        p3 = self.p3_reduce(p3)

        p3 = p3 + p4_up
        p3 = self.p3_refine(p3)

        return p3, p4, p5