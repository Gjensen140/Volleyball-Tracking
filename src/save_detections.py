import os
import pandas as pd

def save_detections_to_csv(label_dir="runs/detect/ball_detection/labels", output_csv="data/detections.csv"):
    rows = []

    for filename in sorted(os.listdir(label_dir)):
        if filename.endswith(".txt"):
            frame_name = filename.replace(".txt", "")  # frame_00001
            with open(os.path.join(label_dir, filename)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        rows.append({
                            "frame": frame_name,
                            "class_id": int(class_id),
                            "x_center": x,
                            "y_center": y,
                            "width": w,
                            "height": h
                        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved {len(df)} detections to {output_csv}")

if __name__ == "__main__":
    save_detections_to_csv()
