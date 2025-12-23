# src/utils/config.py
from pathlib import Path

# Root of the project (change if needed)
PROJECT_ROOT = Path.cwd()  # if running from project root
# If running scripts from other CWDs, set explicit path:
# PROJECT_ROOT = Path("D:/Skin-Lesion-Classification")

# Dataset input path (your provided path)
RAW_DATA_DIR = Path(r"D:\Skin lesion Project\dataset")  # default - edit if needed

# Expected layout in RAW_DATA_DIR:
# - images/   (original images)
# - masks/    (optional: masks that you might have)
# - metadata.csv (optional: csv with image_id, label)
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
RAW_MASKS_DIR = RAW_DATA_DIR / "masks"
METADATA_CSV = RAW_DATA_DIR / "metadata.csv"  # change filename if different

# Processed data output
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_NO_HAIR_DIR = PROCESSED_DIR / "images_no_hair"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

# Model / training config
IMAGE_SIZE = (380, 380)   # B4 recommended: 380x380. Change to (224,224) for B0
BATCH_SIZE = 8            # adjust to fit GPU memory
NUM_CLASSES = 7

EPOCHS = 50
PATIENCE = 10

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
PLOTS_DIR = RESULTS_DIR / "plots"

# Random seed
SEED = 42
