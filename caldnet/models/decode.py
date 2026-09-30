import torch


def decode_boxes(
    predictions: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """
    Decode raw detector predictions into
    image-space bounding boxes.

    Input:

        predictions:
            [B, 4, H, W]

        The four channels represent:

            x
            y
            width
            height

    stride:

        Number of image pixels represented by
        one feature-map cell.

    Returns:

        boxes:
            [B, H, W, 4]

        Box format:

            x1, y1, x2, y2

    Notes:

        This is an educational decoder for CALD-Net v0.1.
        It is NOT yet a reproduction of a specific YOLO
        decoding equation.
    """

    batch_size = predictions.shape[0]
    height = predictions.shape[2]
    width = predictions.shape[3]

    device = predictions.device

    # ---------------------------------------------------------
    # Create grid coordinates
    # ---------------------------------------------------------

    grid_y, grid_x = torch.meshgrid(
        torch.arange(
            height,
            device=device,
            dtype=predictions.dtype,
        ),
        torch.arange(
            width,
            device=device,
            dtype=predictions.dtype,
        ),
        indexing="ij",
    )

    # ---------------------------------------------------------
    # Convert raw x/y predictions into normalized values
    # ---------------------------------------------------------

    center_x = torch.sigmoid(
        predictions[:, 0]
    )

    center_y = torch.sigmoid(
        predictions[:, 1]
    )

    # ---------------------------------------------------------
    # Convert raw width/height into positive values
    # ---------------------------------------------------------

    box_width = torch.exp(
        predictions[:, 2].clamp(
            min=-10.0,
            max=10.0,
        )
    )

    box_height = torch.exp(
        predictions[:, 3].clamp(
            min=-10.0,
            max=10.0,
        )
    )

    # ---------------------------------------------------------
    # Convert cell-relative position to image position
    # ---------------------------------------------------------

    center_x = (
        (grid_x + center_x)
        * stride
    )

    center_y = (
        (grid_y + center_y)
        * stride
    )

    # ---------------------------------------------------------
    # Width and height are currently interpreted
    # in feature-map-cell units.
    # ---------------------------------------------------------

    box_width = box_width * stride
    box_height = box_height * stride

    # ---------------------------------------------------------
    # Convert center-width-height to corners
    # ---------------------------------------------------------

    x1 = (
        center_x
        - box_width / 2
    )

    y1 = (
        center_y
        - box_height / 2
    )

    x2 = (
        center_x
        + box_width / 2
    )

    y2 = (
        center_y
        + box_height / 2
    )

    boxes = torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=-1,
    )

    return boxes