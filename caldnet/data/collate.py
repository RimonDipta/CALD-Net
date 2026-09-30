import torch


def detection_collate_fn(batch):
    """
    Collate function for object detection.

    Each dataset sample contains:

        image:
            [3, H, W]

        targets:
            [N, 5]

    where N can be different for every image.

    Returns:

        images:
            [B, 3, H, W]

        targets:
            list of tensors

    The target list keeps the variable number of
    objects for each image.
    """

    images = []
    targets = []

    for image, target in batch:

        images.append(image)
        targets.append(target)

    images = torch.stack(
        images,
        dim=0,
    )

    return images, targets