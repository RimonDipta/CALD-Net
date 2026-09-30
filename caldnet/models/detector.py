import torch.nn as nn

from .backbone import Backbone
from .neck import FPN
from .head import DetectionHead


class CALDNet(nn.Module):
    """
    CALD-Net v0.1

    Initial baseline architecture:

        Backbone -> FPN -> Multi-scale Detection Head
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.backbone = Backbone()
        self.neck = FPN()
        self.head = DetectionHead(
            num_classes=num_classes,
        )

    def forward(self, x):

        # Feature extraction
        p3, p4, p5 = self.backbone(x)

        # Multi-scale feature fusion
        f3, f4, f5 = self.neck(
            p3,
            p4,
            p5,
        )

        # Detection predictions
        predictions = self.head(
            f3,
            f4,
            f5,
        )

        return predictions