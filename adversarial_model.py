import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchattacks

from model import SimpleCNN


MODEL_FILE = "adversarial_model.pth"


def train_adversarial_model():

    print("[*] Training adversarial model with PGD...")

    device = torch.device("cpu")
    print(f"[*] Using device: {device}")

    transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    attack = torchattacks.PGD(
        model,
        eps=0.3,
        alpha=2 / 255,
        steps=10,
        random_start=True
    )

    model.train()

    for batch_index, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        # Generate PGD adversarial examples
        adversarial_images = attack(images, labels)

        optimizer.zero_grad()

        # Train using adversarial images
        output = model(adversarial_images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        if (batch_index + 1) % 100 == 0:
            print(
                f"[*] Batch {batch_index + 1}/{len(loader)} "
                f"Loss: {loss.item():.4f}"
            )

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    print(f"[*] adversarial model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train_adversarial_model()