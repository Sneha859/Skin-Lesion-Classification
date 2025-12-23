# src/models/train.py
"""
Full training script.
Usage:
python src/models/train.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from src.utils.config import CHECKPOINTS_DIR, RESULTS_DIR, PLOTS_DIR, EPOCHS, BATCH_SIZE, SEED, IMAGE_SIZE, PATIENCE
from src.utils.dataset_loader import get_train_val_test_datasets
from src.models.efficientnet_builder import build_efficientnet_variant
from src.models.model_utils import get_optimizer, get_common_callbacks

import matplotlib.pyplot as plt

def compute_class_weights_from_ds(train_ds, class_names):
    # gather labels
    labels = []
    for batch_x, batch_y in train_ds:
        labels.extend(np.argmax(batch_y.numpy(), axis=1).tolist())
    weights = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=np.array(labels))
    class_weights = {i: float(w) for i, w in enumerate(weights)}
    return class_weights

def plot_history(history, outdir):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title('Accuracy'); plt.legend()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(outdir)/"training_history.png")
    plt.close()

def main():
    Path(CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    # load datasets
    train_ds, val_ds, test_ds, class_names = get_train_val_test_datasets(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    print("Classes:", class_names)

    # compute class weights
    class_weights = compute_class_weights_from_ds(train_ds, class_names)
    print("Class weights:", class_weights)
    # save class weights
    with open(Path(RESULTS_DIR)/"class_weights.json", "w") as f:
        json.dump(class_weights, f)

    # build model (start with B4)
    model = build_efficientnet_variant('B4', input_shape=(*IMAGE_SIZE,3), n_classes=len(class_names))
    model.summary()

    optimizer = get_optimizer('B4')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')])

    checkpoint_path = str(Path(CHECKPOINTS_DIR)/"efficientnet_b4_best.h5")
    callbacks = get_common_callbacks(checkpoint_path, patience=PATIENCE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # save final model
    model.save(Path(CHECKPOINTS_DIR)/"efficientnet_b4_final.h5")
    # plot history
    plot_history(history, PLOTS_DIR)

    # Evaluate on test set
    results = model.evaluate(test_ds)
    print("Test results:", results)

if __name__ == "__main__":
    from pathlib import Path
    main()
