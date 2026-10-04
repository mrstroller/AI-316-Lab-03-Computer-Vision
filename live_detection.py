import os
import time

import cv2
from ultralytics import YOLO


def load_model(path='yolov8n.pt'):
    return YOLO(path)


def draw_fps(frame, fps):
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def save_frame(frame):
    os.makedirs('captures', exist_ok=True)
    filename = time.strftime('captures/%Y%m%d_%H%M%S.jpg')
    cv2.imwrite(filename, frame)
    print('Saved', filename)


def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Cannot open webcam')
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        draw_fps(annotated, fps)

        cv2.imshow('Live Detection', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            save_frame(annotated)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
