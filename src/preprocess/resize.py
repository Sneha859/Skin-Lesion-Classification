# src/preprocess/resize.py
from PIL import Image
import numpy as np

def resize_image(img_rgb, size=(380,380), interpolation=Image.BILINEAR):
    """
    img_rgb: numpy array HxWx3
    returns: resized numpy array
    """
    pil = Image.fromarray(img_rgb)
    pil = pil.resize(size, resample=interpolation)
    return np.array(pil)
