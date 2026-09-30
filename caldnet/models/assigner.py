import torch


class TargetAssigner:
    """
    Simple center-based target assignment
    for CALD-Net v0.1.

    Ground-truth format:

        [x, y, width, height, class_id]

    Coordinates are normalized to [0, 1].
    """

    def __init__(self):
        self.strides = {
            "p3": 8,
            "p4": 16,
            "p5": 32,
        }

    def select_level(
        self,
        width: torch.Tensor,
        height: torch.Tensor,
    ) -> str:
        """
        Select a feature level based on
        approximate normalized object size.
        """

        size = torch.sqrt(
            width * height
        )

        if size < 0.15:
            return "p3"

        if size < 0.30:
            return "p4"

        return "p5"

    def assign(
        self,
        targets: torch.Tensor,
        grid_sizes: dict,
    ):
        """
        targets:

            Tensor [N, 5]

            x, y, w, h, class_id

        grid_sizes:

            {
                "p3": 80,
                "p4": 40,
                "p5": 20,
            }
        """

        assignments = {
            "p3": [],
            "p4": [],
            "p5": [],
        }

        for target in targets:

            x, y, w, h, class_id = target

            level = self.select_level(
                w,
                h,
            )

            grid_size = grid_sizes[level]

            grid_x = min(
                int(x.item() * grid_size),
                grid_size - 1,
            )

            grid_y = min(
                int(y.item() * grid_size),
                grid_size - 1,
            )

            assignments[level].append(
                {
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "box": target[:4],
                    "class_id": int(
                        class_id.item()
                    ),
                }
            )

        return assignments