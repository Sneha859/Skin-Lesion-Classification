import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

from src.utils.config import RAW_IMAGES_DIR, METADATA_CSV, TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, SEED
from src.preprocess.hair_removal import remove_hairs_inpaint_rgb
from src.preprocess.resize import resize_image

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_metadata():
    df = pd.read_csv(METADATA_CSV)

    # one-hot columns
    class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    class_names = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

    def get_label(row):
        values = [row[c] for c in class_cols]
        idx = int(np.argmax(values))
        return class_names[idx]

    df["label"] = df.apply(get_label, axis=1)
    return df[["image", "label"]]

def create_splits():
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(TEST_DIR)

    df = load_metadata()

    # Add .jpg extension
    df["path"] = df["image"].apply(lambda x: RAW_IMAGES_DIR / f"{x}.jpg")

    # Filter missing files just in case
    df = df[df["path"].apply(lambda p: p.exists())]

    # Stratified split
    train_val, test = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=SEED
    )

    train, val = train_test_split(
        train_val, test_size=0.10/0.80, stratify=train_val["label"], random_state=SEED
    )

    def process_set(subset, dest):
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Processing {dest.name}"):
            img_path = row["path"]

            # Load image
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print("Could not read:", img_path)
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Hair removal
            img_clean = remove_hairs_inpaint_rgb(img_rgb)

            # Resize
            img_resized = resize_image(img_clean, size=IMAGE_SIZE)

            # Save output
            class_folder = dest / row["label"]
            ensure_dir(class_folder)

            out_path = class_folder / f"{row['image']}.jpg"
            Image.fromarray(img_resized).save(out_path)

    process_set(train, TRAIN_DIR)
    process_set(val, VAL_DIR)
    process_set(test, TEST_DIR)

    print("Successfully created processed datasets!")
    print("Train:", TRAIN_DIR)
    print("Val:", VAL_DIR)
    print("Test:", TEST_DIR)

if __name__ == "__main__":
    create_splits()
