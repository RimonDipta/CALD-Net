import torch


class TargetBuilder:
    """
    Converts target assignments into dense
    target tensors.

    Each prediction location stores:

        [x, y, w, h, objectness, class0, class1, ...]
    """

    def __init__(
        self,
        num_classes: int,
    ):
        self.num_classes = num_classes

    def build_level(
        self,
        assignments,
        grid_size: int,
        device: torch.device,
    ):
        """
        Build targets for one feature level.

        Output:

            [1, 5 + num_classes, H, W]
        """

        channels = (
            5 + self.num_classes
        )

        target = torch.zeros(
            1,
            channels,
            grid_size,
            grid_size,
            device=device,
        )

        for item in assignments:

            x = item["box"][0]
            y = item["box"][1]
            w = item["box"][2]
            h = item["box"][3]

            grid_x = item["grid_x"]
            grid_y = item["grid_y"]

            class_id = item["class_id"]

            # Bounding box
            target[
                0,
                0,
                grid_y,
                grid_x,
            ] = x

            target[
                0,
                1,
                grid_y,
                grid_x,
            ] = y

            target[
                0,
                2,
                grid_y,
                grid_x,
            ] = w

            target[
                0,
                3,
                grid_y,
                grid_x,
            ] = h

            # Objectness
            target[
                0,
                4,
                grid_y,
                grid_x,
            ] = 1.0

            # One-hot class target
            target[
                0,
                5 + class_id,
                grid_y,
                grid_x,
            ] = 1.0

        return target

    def build(
        self,
        assignments,
        device: torch.device,
    ):
        return {
            "p3": self.build_level(
                assignments["p3"],
                80,
                device,
            ),
            "p4": self.build_level(
                assignments["p4"],
                40,
                device,
            ),
            "p5": self.build_level(
                assignments["p5"],
                20,
                device,
            ),
        }