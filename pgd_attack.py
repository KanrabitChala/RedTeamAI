import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchattacks
from model import SimpleCNN
from PIL import Image


# Change this to test either model
MODEL_FILE = "robust_model.pth"#robust_model.pth

# PGD parameters
EPS = 0.3
ALPHA = 2 / 255
STEPS = 10


def run_pgd_attack_and_save_images():

    print(f"[*] Loading model: {MODEL_FILE}")

    device = torch.device("cpu")

    model = SimpleCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device
        )
    )

    model.eval()

    dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    # Find an image that the clean model correctly classifies
    print("[*] Searching for correctly classified image...")

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_prediction = torch.argmax(
                model(images),
                dim=1
            )

        if clean_prediction.item() == labels.item():
            break

    true_label = labels.item()

    print(f"[*] True Label: {true_label}")
    print(f"[*] Original Model Prediction: {clean_prediction.item()}")

    # Create PGD attack
    attack = torchattacks.PGD(
        model,
        eps=EPS,
        alpha=ALPHA,
        steps=STEPS,
        random_start=True
    )

    print("[*] Generating PGD adversarial example...")

    adversarial_image = attack(
        images,
        labels
    )

    # Predictions
    with torch.no_grad():

        original_prediction = torch.argmax(
            model(images),
            dim=1
        ).item()

        adversarial_prediction = torch.argmax(
            model(adversarial_image),
            dim=1
        ).item()

    # Correct definition of successful evasion
    evasion_success = (
        original_prediction == true_label
        and adversarial_prediction != true_label
    )

    print(f"-> Original Prediction: {original_prediction}")
    print(f"-> PGD Prediction: {adversarial_prediction}")
    print(f"-> Evasion Success: {evasion_success}")

    # Save image
    def tensor_to_image(tensor, filename):

        image_array = (
            tensor
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        if image_array.ndim == 3:
            image_array = image_array.squeeze(0)

        image_array = (
            image_array * 255
        ).clip(0, 255).astype("uint8")

        image = Image.fromarray(image_array)

        image.save(filename)

        print(f"[*] Saved: {filename}")

    tensor_to_image(
        images,
        "pgd_original_sample.png"
    )

    tensor_to_image(
        adversarial_image,
        "pgd_adversarial_sample.png"
    )

    print("[*] PGD attack completed successfully!")


if __name__ == "__main__":
    run_pgd_attack_and_save_images()