# src/models/model_utils.py
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau

def get_optimizer(variant='B4'):
    variant = variant.upper()
    if variant in ('B0','B1','B2','B3','B4','B5'):
        # paper used SGD for smaller models
        opt = optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    else:
        # B6/B7 used Adam in paper for large models
        opt = optimizers.Adam(learning_rate=1e-4)
    return opt

def get_common_callbacks(checkpoint_path, patience=10):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]
    return callbacks
