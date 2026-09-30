import torch

from caldnet.models.assigner import TargetAssigner


def main():

    targets = torch.tensor([
        # x     y     w     h     class
        [0.20, 0.30, 0.10, 0.10, 0],
        [0.60, 0.50, 0.20, 0.20, 1],
        [0.75, 0.70, 0.50, 0.40, 2],
    ])

    assigner = TargetAssigner()

    assignments = assigner.assign(
        targets,
        {
            "p3": 80,
            "p4": 40,
            "p5": 20,
        },
    )

    for level, items in assignments.items():

        print(f"\n{level.upper()}")

        for item in items:
            print(item)


if __name__ == "__main__":
    main()