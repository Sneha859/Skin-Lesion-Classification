# src/models/efficientnet_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils.config import IMAGE_SIZE, NUM_CLASSES

def build_efficientnet_variant(variant='B4', input_shape=None, n_classes=NUM_CLASSES, dropout_rate=0.5):
    """
    variant: 'B0'..'B7' - uses tf.keras.applications
    input_shape: (H,W,3) or None to use IMAGE_SIZE
    """
    if input_shape is None:
        input_shape = (*IMAGE_SIZE, 3)
    variant = variant.upper()
    base = None
    if variant == 'B0':
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    elif variant == 'B1':
        base = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_shape)
    elif variant == 'B2':
        base = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet', input_shape=input_shape)
    elif variant == 'B3':
        base = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape)
    elif variant == 'B4':
        base = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet', input_shape=input_shape)
    elif variant == 'B5':
        base = tf.keras.applications.EfficientNetB5(include_top=False, weights='imagenet', input_shape=input_shape)
    elif variant == 'B6':
        base = tf.keras.applications.EfficientNetB6(include_top=False, weights='imagenet', input_shape=input_shape)
    elif variant == 'B7':
        base = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError("Unsupported variant")

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)

    # paper used two dense blocks for B0-B6; simpler for B7
    if variant != 'B7':
        x = layers.Dense(512, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    else:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model
