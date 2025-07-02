import os
import cv2
import torch
from ultralytics import YOLO

# --- Configuration ---
model_path = "yolov8n.pt"  # Change to your trained YOLOv8 model path
dataset_root = r"H:\PROJECT C\FINAL\dataset_balanced_split"  # Original dataset path
output_root = r"H:\PROJECT C\cropped_faces"  # Output path for cropped faces

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path).to(device)

# --- Ensure Output Root Exists ---
os.makedirs(output_root, exist_ok=True)

def crop_faces_from_dataset():
    """Detects and crops cattle faces using YOLOv8 for all dataset splits."""
    for split in ["train", "test", "valid"]:
        split_path = os.path.join(dataset_root, split)
        output_split_folder = os.path.join(output_root, split)

        if not os.path.exists(split_path):
            print(f"⚠️ Skipping missing directory: {split_path}")
            continue

        for cattle_name in os.listdir(split_path):
            cattle_path = os.path.join(split_path, cattle_name)
            output_cattle_folder = os.path.join(output_split_folder, cattle_name)
            os.makedirs(output_cattle_folder, exist_ok=True)

            for img_name in os.listdir(cattle_path):
                img_path = os.path.join(cattle_path, img_name)
                if not os.path.isfile(img_path):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Cannot read {img_path}. Skipping.")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = model(img_rgb)

                if not results or not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
                    print(f"🚫 No faces detected in {img_path}.")
                    continue

                best_box = results[0].boxes.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = map(int, best_box)
                cropped_face = img[y1:y2, x1:x2]

                if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
                    print(f"⚠️ Invalid crop in {img_path}.")
                    continue

                save_path = os.path.join(output_cattle_folder, f"{os.path.splitext(img_name)[0]}_face.jpg")
                cv2.imwrite(save_path, cropped_face)
                print(f"✅ Saved: {save_path}")

if __name__ == "__main__":
    crop_faces_from_dataset()
