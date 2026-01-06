# src/main.py

from deepfakes import generate_deepfakes, preview_deepfakes
from modelo import train_model
from xai import run_xai
import os

REAL_DIR = "../data/real"
FAKE_DIR = "../data/fake"


def main():
    RUN_DEEPFAKES = False
    PREVIEW_DEEPFAKES = False
    RUN_TRAINING = False
    RUN_XAI = True

    if RUN_DEEPFAKES:
        generate_deepfakes(REAL_DIR, FAKE_DIR, num_deepfakes=6000)

    if PREVIEW_DEEPFAKES:
        preview_deepfakes(REAL_DIR, FAKE_DIR)

    if RUN_TRAINING:
        train_model(REAL_DIR)

    if RUN_XAI:
        run_xai(REAL_DIR, FAKE_DIR)


if __name__ == "__main__":
    main()
