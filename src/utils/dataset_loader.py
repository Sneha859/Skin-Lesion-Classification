import tensorflow as tf
from src.utils.config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE

def get_train_val_test_datasets(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):

    img_size = tuple(image_size)

    # Load datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(TRAIN_DIR),
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(VAL_DIR),
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
        seed=SEED
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(TEST_DIR),
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
        seed=SEED
    )

    # Extract class names BEFORE prefetch
    class_names = train_ds.class_names

    # Add prefetching after extracting class names
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
