import torch
from torch.utils.data import Dataset


class SyntheticDetectionDataset(Dataset):
    """
    Small synthetic object-detection dataset.

    Each sample contains:

        image:
            [3, 640, 640]

        targets:
            [N, 5]

    Target format:

        [x_center, y_center, width, height, class_id]

    Coordinates are normalized to [0, 1].
    """

    def __init__(
        self,
        num_samples: int = 10,
        num_classes: int = 3,
        image_size: int = 640,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        # -----------------------------------------------------
        # Create a synthetic image
        # -----------------------------------------------------

        image = torch.randn(
            3,
            self.image_size,
            self.image_size,
        )

        # -----------------------------------------------------
        # Create synthetic objects
        # -----------------------------------------------------
        #
        # Every sample contains three objects.
        #
        # We deliberately use fixed targets for now so that
        # we can concentrate on understanding the Dataset API.
        # -----------------------------------------------------

        targets = torch.tensor(
            [
                [
                    0.20,
                    0.30,
                    0.10,
                    0.10,
                    0,
                ],
                [
                    0.60,
                    0.50,
                    0.20,
                    0.20,
                    1,
                ],
                [
                    0.75,
                    0.70,
                    0.50,
                    0.40,
                    2,
                ],
            ],
            dtype=torch.float32,
        )

        return image, targets