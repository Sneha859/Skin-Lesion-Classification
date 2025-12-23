# src/preprocess/hair_removal.py
import cv2
import numpy as np

def remove_hairs_inpaint_rgb(img_rgb, kernel_size=(9,9), threshold=10, dilate_iter=1):
    """
    Remove hair-like dark lines using blackhat morphological operation then inpaint (Telea).
    img_rgb: numpy array in RGB (H,W,3)
    returns: inpainted rgb image
    """
    # convert to gray
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # threshold to create mask of hairs
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    # dilate to make sure hair is covered
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=dilate_iter)
    # inpaint on original RGB image (note cv2 expects BGR, but inpaint works on channels; convert to BGR temporarily)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    inpaint = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    inpaint_rgb = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
    return inpaint_rgb
