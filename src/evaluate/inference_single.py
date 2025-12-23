# src/evaluate/inference_single.py
"""
Predict single image using the best checkpoint.
Usage:
python src/evaluate/inference_single.py --image "path/to/image.jpg"
"""

import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.utils.config import CHECKPOINTS_DIR, IMAGE_SIZE
from src.utils.dataset_loader import get_train_val_test_datasets

def load_image(path, size):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    arr = np.array(img) / 255.0
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    model_path = Path(CHECKPOINTS_DIR)/"efficientnet_b4_best.h5"
    if not model_path.exists():
        model_path = Path(CHECKPOINTS_DIR)/"efficientnet_b4_final.h5"
    model = tf.keras.models.load_model(model_path)
    _, _, test_ds, class_names = get_train_val_test_datasets()
    img_arr = load_image(args.image, IMAGE_SIZE)
    x = np.expand_dims(img_arr, axis=0)
    preds = model.predict(x)
    top_idx = np.argmax(preds[0])
    print("Predicted:", class_names[top_idx], "Prob:", float(preds[0, top_idx]))
    # print top-3
    top3 = preds[0].argsort()[-3:][::-1]
    print("Top-3 predictions:")
    for idx in top3:
        print(class_names[idx], float(preds[0, idx]))

if __name__ == "__main__":
    main()
