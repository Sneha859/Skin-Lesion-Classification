# app_streamlit.py
"""
Streamlit app for Skin Lesion Classification (EfficientNet B4).
Save this file in your project root (same folder that contains `src/`, `checkpoints/`, etc).
Run: streamlit run app_streamlit.py
"""

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import io
import os
import sys
import tensorflow as tf

# --- Ensure project root is on sys.path so imports like `from src.utils...` work ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import your project utilities/config
try:
    from src.utils.config import CHECKPOINTS_DIR, IMAGE_SIZE
    from src.utils.dataset_loader import get_train_val_test_datasets
except Exception as e:
    st.error(f"Could not import project modules: {e}")
    st.stop()

# ---------------------------
# Model loading utility
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    ckpt_dir = Path(CHECKPOINTS_DIR)
    best = ckpt_dir / "efficientnet_b4_best.h5"
    final = ckpt_dir / "efficientnet_b4_final.h5"
    model_path = best if best.exists() else final if final.exists() else None
    if model_path is None:
        raise FileNotFoundError(f"No model found in {ckpt_dir}. Expected efficientnet_b4_best.h5 or efficientnet_b4_final.h5")
    # Use compile=False to speed up loading if not using .fit further
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, str(model_path)

# ---------------------------
# Helper: preprocess
# ---------------------------
def preprocess_pil_image(pil_img, size):
    img = pil_img.convert("RGB")
    img = img.resize(tuple(size))
    arr = np.array(img).astype(np.float32) / 255.0
    # model expects shape (1, H, W, 3)
    return np.expand_dims(arr, axis=0)

# ---------------------------
# Helper: Grad-CAM
# ---------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    img_array: input (1,H,W,3), preprocessed (0-1)
    model: keras model
    last_conv_layer_name: optional str. If None, find automatically.
    returns heatmap (H,W) normalized to [0,1]
    """
    # find last conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("Could not find a 4D conv layer in the model to compute Grad-CAM.")

    last_conv_layer = model.get_layer(last_conv_layer_name)
    # Create a model that maps the input image to the activations of the last conv layer
    # and the model's predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        pred = predictions[:, pred_index]

    # compute gradients of the predicted class w.r.t. conv layer outputs
    grads = tape.gradient(pred, conv_outputs)
    # compute channel-wise mean of gradients (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (img_array.shape[1], img_array.shape[2]))[..., 0]
    heatmap = tf.clip_by_value(heatmap, 0, 1).numpy()
    return heatmap

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")
st.title("Skin Lesion Classification — Demo")

# Load model and class names
with st.spinner("Loading model and class names (this may take a few seconds)..."):
    try:
        model, model_path = load_model()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

    # get class names using dataset loader (fast: it just reads directories)
    try:
        # call the loader with default params just to get class_names
        _, _, _, class_names = get_train_val_test_datasets(image_size=IMAGE_SIZE, batch_size=1)
    except Exception:
        # fallback: try reading from train folder structure
        train_dir = Path("dataset") / "train"
        if train_dir.exists():
            class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        else:
            # if everything fails, make dummy
            class_names = [f"class_{i}" for i in range(model.output_shape[-1])]
st.success(f"Model loaded from: `{model_path}`")
st.write(f"Classes: {class_names}")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    st.write("Or choose a sample from dataset:")
    sample_select = st.selectbox("Sample image", ["--none--"] + sorted([p.name for p in (PROJECT_ROOT / 'dataset' / 'images').glob("*.jpg")] ) if (PROJECT_ROOT / 'dataset' / 'images').exists() else ["--none--"])

    if st.button("Run prediction"):
        if uploaded is None and sample_select in ("--none--",):
            st.error("Please upload an image or pick a sample.")
        else:
            # get PIL image
            if uploaded is not None:
                image_data = uploaded.read()
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
            else:
                img = Image.open(PROJECT_ROOT / 'dataset' / 'images' / sample_select).convert("RGB")

            # preprocess
            x = preprocess_pil_image(img, IMAGE_SIZE)

            # predict
            preds = model.predict(x)
            probs = preds[0]
            top_idx = np.argmax(probs)
            top3_idx = probs.argsort()[-3:][::-1]

            st.image(img, caption="Input image", use_column_width=True)

            st.subheader("Predictions")
            st.write(f"Top-1: **{class_names[top_idx]}** — probability: **{float(probs[top_idx]):.4f}**")

            # display top-3 table
            import pandas as pd
            rows = []
            for i in top3_idx:
                rows.append({"class": class_names[i], "prob": float(probs[i])})
            df = pd.DataFrame(rows)
            st.table(df)

            # Grad-CAM
            st.subheader("Grad-CAM Heatmap")
            with st.spinner("Computing Grad-CAM..."):
                try:
                    heatmap = make_gradcam_heatmap(x, model)
                    # convert heatmap to RGBA overlay
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap("jet")
                    heatmap_rgb = cmap(heatmap)[:, :, :3]  # drop alpha
                    heatmap_img = Image.fromarray((heatmap_rgb * 255).astype("uint8")).resize(img.size)
                    # overlay: blend original and heatmap
                    overlay = Image.blend(img, heatmap_img, alpha=0.4)
                    st.image(overlay, use_column_width=True, caption="Grad-CAM overlay (alpha=0.4)")
                except Exception as e:
                    st.warning(f"Grad-CAM failed: {e}")

with col2:
    st.markdown("### Instructions")
    st.write("""
    - Upload a skin lesion image (jpg/png) or select a sample image shipped in `dataset/images`.
    - The app will load the model from `checkpoints/efficientnet_b4_best.h5` (or `efficientnet_b4_final.h5`).
    - Top-1 and Top-3 predictions are shown, plus a Grad-CAM heatmap overlay.
    """)
    st.markdown("### Notes")
    st.write("""
    - If your model or dataset folders are in different locations, edit `CHECKPOINTS_DIR` or `dataset/images`.
    - If the app cannot import `src`, be sure the app file is saved in the project root (same folder that contains `src/`) or add the project path to `PYTHONPATH`.
    - Run Streamlit in the same virtual environment where TensorFlow GPU is installed.
    """)

st.markdown("---")
st.caption("Streamlit demo — Skin Lesion Classification (EfficientNet-B4)")
