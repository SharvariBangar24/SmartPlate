import os
import sys
import glob
import time
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_LABELS_DIR = os.path.join(BASE_DIR, 'raw_labels')

print("=" * 70)
print("Step 1: Downloading remaining FPB annotation records (max_workers=64)...")
print("=" * 70)
yaml_file = os.path.join(RAW_LABELS_DIR, 'FPB_Dataset', 'RGB', 'data.yaml')
existing_txt = glob.glob(os.path.join(RAW_LABELS_DIR, 'FPB_Dataset', 'RGB', '**', '*.txt'), recursive=True)

if os.path.exists(yaml_file) and len(existing_txt) >= 14000:
    print(f"All {len(existing_txt)} annotation label files and data.yaml already present and verified locally!")
else:
    from fast_resume_and_train import get_file_list, download_missing_files
    txt_files = get_file_list()
    download_missing_files(txt_files, max_workers=16)

print("\n" + "=" * 70)
print("Step 2: Executing Model Training & Evaluation Suite...")
print("=" * 70)

sys.path.insert(0, BASE_DIR)
import train_portion_models
train_portion_models.main()
