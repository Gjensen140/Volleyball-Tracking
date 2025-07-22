import cv2
import pandas as pd
import os

def draw_detections(csv_file, frames_dir, output_dir="output/annotated_frames"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        frame_file = f"frame_{int(row['frame']):05d}.jpg"
        frame_path = os.path.join(frames_dir, frame_file)

        if not os.path.exists(frame_path):
            continue

        frame = cv2.imread(frame_path)
        h, w, _ = frame.shape

        # Convert YOLO format to pixel coordinates
        x_center = int(row['x_center'] * w)
        y_center = int(row['y_center'] * h)
        box_w = int(row['width'] * w)
        box_h = int(row['height'] * h)

        x1 = x_center - box_w // 2
        y1 = y_center - box_h // 2
        x2 = x_center + box_w // 2
        y2 = y_center + box_h // 2

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, frame_file), frame)

    print(f"[INFO] Annotated frames saved to {output_dir}")
