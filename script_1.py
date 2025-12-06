import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "traffic.mp4"
LINE_Y = 400
CONF_THRESHOLD = 0.4
DENSITY_WINDOW = 30

model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = ["car", "motorbike", "bus", "truck"]

def is_vehicle(class_name):
    return class_name in VEHICLE_CLASSES

def draw_text(img, text, pos, scale=0.7, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thickness, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: could not open video.")
        return

    vehicle_count = 0
    recent_counts = []
    track_ids_seen = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 0), 2)

        results = model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
        current_frame_vehicles = 0

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    if not is_vehicle(class_name):
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    track_id = int(box.id[0]) if box.id is not None else None

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    label = f"{class_name}" + (f" #{track_id}" if track_id is not None else "")
                    draw_text(frame, label, (x1, y1 - 5), scale=0.5, thickness=1)

                    current_frame_vehicles += 1

                    if abs(cy - LINE_Y) < 10 and track_id is not None:
                        if track_id not in track_ids_seen:
                            track_ids_seen.add(track_id)
                            vehicle_count += 1

        recent_counts.append(current_frame_vehicles)
        if len(recent_counts) > DENSITY_WINDOW:
            recent_counts.pop(0)

        avg_density = np.mean(recent_counts) if recent_counts else 0

        draw_text(frame, f"Total Vehicles: {vehicle_count}", (20, 40))
        draw_text(frame, f"Current Frame Vehicles: {current_frame_vehicles}", (20, 70))
        draw_text(frame, f"Traffic Density (avg last {DENSITY_WINDOW} frames): {avg_density:.1f}", (20, 100))

        cv2.imshow("Vehicle Counting & Traffic Density", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Total Vehicles Counted: {vehicle_count}")

if __name__ == "__main__":
    main()
