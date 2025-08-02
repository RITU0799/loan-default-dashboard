import os
import subprocess
import joblib

MODEL_IDS = {
    "gbm_pipeline.pkl": "1lkGXtpnsbiPaelJ-AYwQqTwkKg_Ku6Kk",
    "model_columns.pkl": "11Mzz8mDmaBVAtgK2WKafUxOpph6xMeU3",
    "rf_model.pkl": "19NV4y4OVwardcBmrZ3UT9SaLGR2aPyqP"
}
MODEL_DIR = "models"

def download_if_missing():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, file_id in MODEL_IDS.items():
        target = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(target):
            print(f"Downloading {fname}...")
            # install gdown if not present
            try:
                import gdown
            except ImportError:
                subprocess.run(["pip", "install", "gdown"], check=True)
                import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, target, quiet=False)

def load_model(path):
    return joblib.load(path)

def ensure_all_models():
    download_if_missing()

