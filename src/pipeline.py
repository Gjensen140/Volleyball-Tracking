from detect_ball import detect_volleyball
from save_detections import save_detections_to_csv

def run_pipeline():
    print("[1] Running YOLOv8 detection on frames...")
    detect_volleyball()

    print("[2] Exporting detections to CSV...")
    save_detections_to_csv()

    print("[✅] Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
