from ultralytics import YOLO
import shutil
import os

def train_yolov8(
    model_arch="yolov8n.yaml",         # you can also try yolov8s.yaml
    data_yaml="data/data.yaml",
    epochs=50,
    imgsz=640,
    save_path="models/yolov8_ball.pt"
):
    print("[INFO] Starting training...")
    model = YOLO(model_arch)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

    # After training, copy best.pt to your models directory
    best_weight_path = "runs/detect/train/weights/best.pt"
    if os.path.exists(best_weight_path):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_weight_path, save_path)
        print(f"[INFO] Model saved to {save_path}")
    else:
        print("[ERROR] Training completed but best.pt not found.")

if __name__ == "__main__":
    train_yolov8()
