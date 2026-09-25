import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchattacks
from model import SimpleCNN
from PIL import Image

def run_attack_and_save_images():
    print("[*] Loading model for adversarial attack...")
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Load MNIST test dataset
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    images, labels = next(iter(loader))

    true_label = labels.item()
    print(f"[*] True Label: {true_label}")

    # Generate FGSM adversarial attack
    eps_val = 0.3
    atk = torchattacks.FGSM(model, eps=eps_val)
    adv_image = atk(images, labels)

    # Get model predictions
    orig_pred = torch.argmax(model(images)).item()
    adv_pred = torch.argmax(model(adv_image)).item()

    print(f"-> Original Model Prediction: {orig_pred}")
    print(f"-> Adversarial Model Prediction (FGSM): {adv_pred} (Evasion Success: {orig_pred != adv_pred})")

    # Helper function to convert a PyTorch tensor back into a viewable image file
    def tensor_to_image(tensor_tensor, filename):
        # Remove batch dimension [1, 1, 28, 28] -> [1, 28, 28], move to CPU, convert to numpy
        img_np = tensor_tensor.squeeze(0).detach().cpu().numpy()
        # Shape is [channels, height, width] -> transpose to [height, width, channels] if needed, but for grayscale [1, 28, 28] -> [28, 28]
        if img_np.ndim == 3:
            img_np = img_np.squeeze(0)
        
        # Scale pixel values from [0, 1] back to [0, 255]
        img_np = (img_np * 255).astype('uint8')
        
        # Save using PIL
        img_pil = Image.fromarray(img_np)
        img_pil.save(filename)
        print(f"[*] Saved image to: {filename}")

    # Save both images to your project folder
    tensor_to_image(images, "original_sample.png")
    tensor_to_image(adv_image, "adversarial_sample.png")
    print("[*] Success!")

if __name__ == "__main__":
    run_attack_and_save_images()