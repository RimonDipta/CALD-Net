import torch

from caldnet.models.detector import CALDNet


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = CALDNet(
        num_classes=3
    ).to(device)

    model.eval()

    x = torch.randn(
        1,
        3,
        640,
        640,
        device=device,
    )

    with torch.no_grad():
        predictions = model(x)

    print("\nModel output:")

    for level, prediction in predictions.items():

        print(
            f"{level.upper()}:",
            prediction.shape,
            prediction.device,
        )


if __name__ == "__main__":
    main()