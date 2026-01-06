# src/deepfakes.py

import random
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageChops, ImageFilter
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "../outputs/fase1_preview"
os.makedirs(OUTPUT_DIR, exist_ok=True)


PROMPT = (
    "realistic human face, same identity, subtle but noticeable changes in facial features, "
    "slightly different eyes shape, nose proportions or lips contour, natural skin texture, "
    "realistic lighting, high detail"
)

NEGATIVE_PROMPT = (
    "cartoon, anime, painting, blur, distortion, deformed face, extra eyes, extra mouth, bad anatomy"
)


def facial_regions(w, h):
    return {
        "eyes": (int(0.20*w), int(0.28*h), int(0.80*w), int(0.48*h)),
        "nose": (int(0.38*w), int(0.45*h), int(0.62*w), int(0.70*h)),
        "mouth": (int(0.32*w), int(0.65*h), int(0.68*w), int(0.82*h)),
    }


def create_combined_mask(image):
    w, h = image.size
    regions = facial_regions(w, h)
    chosen = random.sample(list(regions.keys()), random.randint(2, 3))

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for r in chosen:
        draw.rectangle(regions[r], fill=255)

    return mask.filter(ImageFilter.GaussianBlur(radius=6))


def visualize(real, mask, fake, idx):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(real)
    plt.axis("off")
    plt.title("Real")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.title("Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(fake)
    plt.axis("off")
    plt.title("Fake")

    save_path = os.path.join(OUTPUT_DIR, f"preview_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_pipe(device):
    return StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)


def generate_deepfakes(real_dir, fake_dir, num_deepfakes=6000, device=None):
    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    fake_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipe(device)

    all_files = sorted(real_dir.glob("*"))
    selected = random.sample(all_files, num_deepfakes)

    for i, img_path in enumerate(selected):
        print(f"[{i+1}/{num_deepfakes}] {img_path.name}", end="\r")

        image = Image.open(img_path).convert("RGB")
        mask = create_combined_mask(image)

        for _ in range(2):
            with torch.no_grad():
                fake = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=image,
                    mask_image=mask,
                    num_inference_steps=35,
                    guidance_scale=9.0
                ).images[0]

            diff = np.mean(np.array(ImageChops.difference(image, fake)))
            if diff > 6.0:
                break

        fake.save(fake_dir / img_path.name)

    print("\n[Deepfakes] Concluído.")


def preview_deepfakes(real_dir, fake_dir, num_preview=4):
    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)

    fake_files = sorted(fake_dir.glob("*"))
    preview = random.sample(fake_files, min(num_preview, len(fake_files)))

    for idx, f in enumerate(preview):
        real = Image.open(real_dir / f.name).convert("RGB")
        fake = Image.open(f).convert("RGB")
        mask = create_combined_mask(real)

        visualize(real, mask, fake, idx)
