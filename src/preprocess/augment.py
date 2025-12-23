# src/preprocess/augment.py
import albumentations as A

def get_train_augmentations(image_size=(380,380)):
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.3),
        A.Resize(*image_size)
    ])

def get_val_augmentations(image_size=(380,380)):
    return A.Compose([
        A.Resize(*image_size)
    ])
