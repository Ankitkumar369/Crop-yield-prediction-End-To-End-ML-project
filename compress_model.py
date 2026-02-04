import joblib
import os

old_file = "model_crop.pkl"
new_file = "model_crop_compressed.pkl"

print("Original size:",
      os.path.getsize(old_file) / (1024*1024), "MB")

obj = joblib.load(old_file)

joblib.dump(obj, new_file, compress=9)

print("Compressed size:",
      os.path.getsize(new_file) / (1024*1024), "MB")
