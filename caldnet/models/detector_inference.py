import torch

from .inference import (
    decode_predictions,
    filter_predictions,
)

from .nms import nms


def run_inference(
    predictions: dict,
    num_classes: int,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
):
    """
    Run complete multi-scale inference.

    Parameters
    ----------
    predictions:
        Dictionary containing:

            p3
            p4
            p5

        Each tensor has shape:

            [B, 5 + num_classes, H, W]

    num_classes:
        Number of object classes.

    confidence_threshold:
        Minimum confidence required before NMS.

    iou_threshold:
        IoU threshold used by NMS.

    Returns
    -------
    detections:
        List containing one dictionary per image.

        Each dictionary contains:

            boxes
            scores
            class_ids
    """

    strides = {
        "p3": 8,
        "p4": 16,
        "p5": 32,
    }

    # ---------------------------------------------------------
    # We currently handle one image at a time.
    # ---------------------------------------------------------

    batch_size = predictions["p3"].shape[0]

    results = []

    # ---------------------------------------------------------
    # Process every image in the batch
    # ---------------------------------------------------------

    for image_index in range(
        batch_size
    ):

        all_boxes = []
        all_scores = []
        all_class_ids = []

        # -----------------------------------------------------
        # Process P3, P4 and P5
        # -----------------------------------------------------

        for level in [
            "p3",
            "p4",
            "p5",
        ]:

            prediction = predictions[
                level
            ][
                image_index:image_index + 1
            ]

            # -------------------------------------------------
            # Decode
            # -------------------------------------------------

            decoded = decode_predictions(
                prediction,
                stride=strides[level],
                num_classes=num_classes,
            )

            # -------------------------------------------------
            # Confidence filtering
            # -------------------------------------------------

            (
                boxes,
                scores,
                class_ids,
            ) = filter_predictions(
                decoded,
                confidence_threshold=confidence_threshold,
            )

            # -------------------------------------------------
            # Store candidates
            # -------------------------------------------------

            if boxes.numel() > 0:

                all_boxes.append(
                    boxes
                )

                all_scores.append(
                    scores
                )

                all_class_ids.append(
                    class_ids
                )

        # -----------------------------------------------------
        # No detections
        # -----------------------------------------------------

        if len(all_boxes) == 0:

            results.append(
                {
                    "boxes": torch.empty(
                        0,
                        4,
                        device=predictions[
                            "p3"
                        ].device,
                    ),
                    "scores": torch.empty(
                        0,
                        device=predictions[
                            "p3"
                        ].device,
                    ),
                    "class_ids": torch.empty(
                        0,
                        dtype=torch.long,
                        device=predictions[
                            "p3"
                        ].device,
                    ),
                }
            )

            continue

        # -----------------------------------------------------
        # Combine all feature levels
        # -----------------------------------------------------

        all_boxes = torch.cat(
            all_boxes,
            dim=0,
        )

        all_scores = torch.cat(
            all_scores,
            dim=0,
        )

        all_class_ids = torch.cat(
            all_class_ids,
            dim=0,
        )

        # -----------------------------------------------------
        # Class-aware NMS
        # -----------------------------------------------------

        final_indices = []

        for class_id in range(
            num_classes
        ):

            class_mask = (
                all_class_ids
                == class_id
            )

            class_indices = (
                torch.nonzero(
                    class_mask,
                    as_tuple=False,
                )
                .squeeze(1)
            )

            if class_indices.numel() == 0:
                continue

            class_boxes = all_boxes[
                class_indices
            ]

            class_scores = all_scores[
                class_indices
            ]

            keep = nms(
                class_boxes,
                class_scores,
                iou_threshold=iou_threshold,
            )

            final_indices.append(
                class_indices[
                    keep
                ]
            )

        # -----------------------------------------------------
        # Combine class-specific results
        # -----------------------------------------------------

        if len(final_indices) > 0:

            final_indices = torch.cat(
                final_indices,
                dim=0,
            )

            # Sort final detections by confidence
            order = torch.argsort(
                all_scores[
                    final_indices
                ],
                descending=True,
            )

            final_indices = (
                final_indices[
                    order
                ]
            )

        else:

            final_indices = torch.empty(
                0,
                dtype=torch.long,
                device=all_boxes.device,
            )

        # -----------------------------------------------------
        # Final results
        # -----------------------------------------------------

        results.append(
            {
                "boxes": all_boxes[
                    final_indices
                ],
                "scores": all_scores[
                    final_indices
                ],
                "class_ids": all_class_ids[
                    final_indices
                ],
            }
        )

    return results