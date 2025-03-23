import gdown
import os

# Google Drive file ID
file_id = "dyEekxHzR0Nyod9Qgn4DGA6tQXfKfGaks"  # Replace with your actual file ID

# Path to store the model
model_dir = os.path.join("model")
os.makedirs(model_dir, exist_ok=True)

output = os.path.join(model_dir, "xgboost_model.pkl")

# Download the model if it doesn't exist
if not os.path.exists(output):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    print("Model downloaded successfully!")
else:
    print("Model already exists.")#https://drive.google.com/file/dyEekxHzR0Nyod9Qgn4DGA6tQXfKfGaks/1/view?usp=sharing