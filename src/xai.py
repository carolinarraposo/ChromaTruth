# src/xai.py
import os
import csv
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from modelo import UNet, DEVICE, IMG_SIZE, MODEL_SAVE_PATH

OUTPUT_DIR = "../outputs/fase3_xai"
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")
REAL_DIR_OUT = os.path.join(MAPS_DIR, "real")
FAKE_DIR_OUT = os.path.join(MAPS_DIR, "fake")

os.makedirs(REAL_DIR_OUT, exist_ok=True)
os.makedirs(FAKE_DIR_OUT, exist_ok=True)


def compute_error_maps(L, ab_real, ab_pred):
    error_map = torch.sqrt(torch.sum((ab_real - ab_pred) ** 2, dim=0)).cpu().numpy()
    norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    smooth = cv2.GaussianBlur(norm, (9, 9), 0)

    heat = (smooth * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    L_np = (L.squeeze().cpu().numpy() * 100).astype(np.float64)
    lab = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    lab[:, :, 0] = L_np
    bg = (color.lab2rgb(lab) * 255).astype(np.uint8)

    overlap = cv2.addWeighted(bg, 0.6, heat, 0.4, 0)

    return error_map, smooth, bg, overlap


class ChromaTruthDetector:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.model = UNet().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        lab = color.rgb2lab(np.array(img) / 255.0)

        L = torch.tensor(lab[:, :, 0:1] / 100.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        ab = torch.tensor(lab[:, :, 1:] / 128.0).permute(2, 0, 1).float().to(DEVICE)

        with torch.no_grad():
            pred = self.model(L).squeeze(0)

        error_map, smooth, bg, overlap = compute_error_maps(L, ab, pred)
        score = float(smooth.mean())

        return score, bg, overlap, error_map, smooth


def evaluate_and_save(real_dir, fake_dir, detector):
    scores, labels, names = [], [], []

    for label_dir, label, out_dir in [
        (real_dir, 0, REAL_DIR_OUT),
        (fake_dir, 1, FAKE_DIR_OUT)
    ]:
        files = sorted(Path(label_dir).glob("*"))

        for f in tqdm(files, desc=f"Processando {'REAL' if label==0 else 'FAKE'}"):
            score, bg, overlap, *_ = detector.predict(f)

            name = f.stem
            cv2.imwrite(os.path.join(out_dir, f"{name}_overlap.png"), overlap[:, :, ::-1])
            cv2.imwrite(os.path.join(out_dir, f"{name}_bg.png"), bg[:, :, ::-1])

            scores.append(score)
            labels.append(label)
            names.append(f.name)

    return np.array(scores), np.array(labels), names


def compute_roc(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    idx = np.argmax(tpr - fpr)
    return fpr, tpr, thr, auc_val, idx


def run_xai(real_dir, fake_dir):
    detector = ChromaTruthDetector()

    scores, labels, names = evaluate_and_save(real_dir, fake_dir, detector)
    fpr, tpr, thr, auc_val, idx = compute_roc(scores, labels)

    threshold = thr[idx]

    print(f"AUC = {auc_val:.4f}")
    print(f"Threshold ótimo = {threshold:.4f}")

    # ROC curve
    plt.plot(fpr, tpr)
    plt.scatter(fpr[idx], tpr[idx])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – ChromaTruth")
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    with open(os.path.join(OUTPUT_DIR, "scores.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "score"])
        for n, l, s in zip(names, labels, scores):
            writer.writerow([n, "FAKE" if l else "REAL", s])

    return detector, threshold
