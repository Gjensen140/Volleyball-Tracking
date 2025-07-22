from ultralytics import YOLO

def train_yolov8(model_size="yolov8n.yaml", data_yaml="data/data.yaml", epochs=50, imgsz=640):
    model = YOLO(model_size)  # Can also be 'yolov8n.pt' to fine-tune
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

if __name__ == "__main__":
    train_yolov8()
