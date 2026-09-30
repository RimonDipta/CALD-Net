import torch


class TargetBuilder:
    """
    Batch-aware target builder for CALD-Net.

    Each object target is represented as:

        [x_center, y_center, width, height, class_id]

    Coordinates are normalized to [0, 1].

    Dense target representation:

        [x, y, w, h, objectness, class0, class1, ...]

    Shape for one feature level:

        [B, 5 + num_classes, H, W]
    """

    def __init__(
        self,
        num_classes: int,
        grid_sizes: dict | None = None,
    ):
        self.num_classes = num_classes

        if grid_sizes is None:
            grid_sizes = {
                "p3": 80,
                "p4": 40,
                "p5": 20,
            }

        self.grid_sizes = grid_sizes

    def build_level(
        self,
        batch_assignments,
        grid_size: int,
        batch_size: int,
        device: torch.device,
    ):
        """
        Build targets for one feature level.

        Args:
            batch_assignments:
                List containing assignments for each image.

            grid_size:
                Spatial size of this feature level.

            batch_size:
                Number of images in the batch.

            device:
                Target tensor device.

        Returns:
            Tensor:
                [B, 5 + num_classes, grid_size, grid_size]
        """

        channels = 5 + self.num_classes

        target = torch.zeros(
            batch_size,
            channels,
            grid_size,
            grid_size,
            device=device,
            dtype=torch.float32,
        )

        for batch_index, assignments in enumerate(
            batch_assignments
        ):

            for item in assignments:

                x = item["box"][0]
                y = item["box"][1]
                w = item["box"][2]
                h = item["box"][3]

                grid_x = item["grid_x"]
                grid_y = item["grid_y"]
                class_id = item["class_id"]

                target[
                    batch_index,
                    0,
                    grid_y,
                    grid_x,
                ] = x

                target[
                    batch_index,
                    1,
                    grid_y,
                    grid_x,
                ] = y

                target[
                    batch_index,
                    2,
                    grid_y,
                    grid_x,
                ] = w

                target[
                    batch_index,
                    3,
                    grid_y,
                    grid_x,
                ] = h

                target[
                    batch_index,
                    4,
                    grid_y,
                    grid_x,
                ] = 1.0

                target[
                    batch_index,
                    5 + class_id,
                    grid_y,
                    grid_x,
                ] = 1.0

        return target

    def build_batch(
        self,
        batch_assignments,
        device: torch.device,
    ):
        """
        Build dense targets for an entire batch.

        Args:
            batch_assignments:
                Dictionary:

                    {
                        "p3": [
                            assignments_image_0,
                            assignments_image_1,
                            ...
                        ],
                        "p4": [...],
                        "p5": [...]
                    }

        Returns:
            Dictionary:

                {
                    "p3": [B,C,80,80],
                    "p4": [B,C,40,40],
                    "p5": [B,C,20,20]
                }
        """

        batch_size = len(
            batch_assignments["p3"]
        )

        return {
            level: self.build_level(
                batch_assignments=batch_assignments[
                    level
                ],
                grid_size=self.grid_sizes[level],
                batch_size=batch_size,
                device=device,
            )
            for level in [
                "p3",
                "p4",
                "p5",
            ]
        }

    def build(
        self,
        assignments,
        device: torch.device,
    ):
        """
        Backward-compatible single-image builder.

        This method accepts:

            {
                "p3": [...],
                "p4": [...],
                "p5": [...]
            }

        and returns tensors with batch dimension 1.
        """

        batch_assignments = {
            "p3": [
                assignments["p3"]
            ],
            "p4": [
                assignments["p4"]
            ],
            "p5": [
                assignments["p5"]
            ],
        }

        return self.build_batch(
            batch_assignments=batch_assignments,
            device=device,
        )