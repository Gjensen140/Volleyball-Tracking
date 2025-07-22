from ultralytics import YOLO
import os

def detect_volleyball(model_path="models/yolov8_ball.pt", input_path="data/volleyball/images/val", output_dir="runs/detect"):
    model = YOLO(model_path)

    results = model.predict(
        source=input_path,     # Folder of frames
        conf=0.25,
        save=True,
        save_txt=True,
        project=output_dir,
        name="ball_detection"
    )

    return results

if __name__ == "__main__":
    detect_volleyball()
