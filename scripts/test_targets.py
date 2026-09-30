import torch

from caldnet.models.assigner import TargetAssigner
from caldnet.models.targets import TargetBuilder


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    targets = torch.tensor(
        [
            # x     y     w     h     class
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
        device=device,
    )

    assigner = TargetAssigner()

    assignments = assigner.assign(
        targets,
        {
            "p3": 80,
            "p4": 40,
            "p5": 20,
        },
    )

    builder = TargetBuilder(
        num_classes=3
    )

    target_tensors = builder.build(
        assignments,
        device,
    )

    print("\nTarget tensors:")

    for level, tensor in target_tensors.items():

        print(
            f"{level.upper()}:",
            tensor.shape,
            tensor.device,
        )

        object_locations = torch.nonzero(
            tensor[0, 4] == 1
        )

        print(
            "Positive locations:",
            object_locations,
        )


if __name__ == "__main__":
    main()