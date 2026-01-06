# src/modelo.py

import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
MODEL_SAVE_PATH = Path("../models/unet_chroma_truth.pth")

OUTPUT_DIR = "../outputs/fase2_treino"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FaceColorDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted([
            Path(root_dir) / f for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)

        lab = color.rgb2lab(img.permute(1, 2, 0).numpy())
        L = torch.tensor(lab[:, :, 0:1] / 100.0).permute(2, 0, 1).float()
        ab = torch.tensor(lab[:, :, 1:] / 128.0).permute(2, 0, 1).float()

        return L, ab


def create_dataloaders(real_dir, batch_size=16, num_workers=0):
    dataset = FaceColorDataset(real_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(DEVICE == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(DEVICE == "cuda"))

    return train_loader, val_loader


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        c4 = self.down4(self.pool(c3))

        u3 = self.conv3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), c1], dim=1))

        return self.final(u1)


def train_model(real_dir, epochs=15, lr=2e-4, batch_size=16, num_workers=0):
    train_loader, val_loader = create_dataloaders(real_dir, batch_size, num_workers)

    model = UNet().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    patience = 3
    wait = 0

    train_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for L, ab in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            L, ab = L.to(DEVICE), ab.to(DEVICE)

            optimizer.zero_grad()
            pred = model(L)
            loss = criterion(pred, ab)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(DEVICE), ab.to(DEVICE)
                val_loss += criterion(model(L), ab).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        print(f"[{epoch+1}] Train={train_loss:.4f} | Val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("[Checkpoint] Modelo salvo.")
        else:
            wait += 1
            if wait >= patience:
                print("[Early Stopping]")
                break

    plt.plot(train_hist, label="Treino")
    plt.plot(val_hist, label="Validação")
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return model
