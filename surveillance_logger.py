import os
import time

import cv2
from ultralytics import YOLO

TARGET_CLASSES = ['person', 'cell phone']
THRESHOLD = 0.65
COOLDOWN = 2


def setup():
    os.makedirs('alerts', exist_ok=True)
    if not os.path.exists('events.log'):
        with open('events.log', 'w') as f:
            f.write('Timestamp,Class,Confidence,X1,Y1,X2,Y2\n')


def get_events(result, names):
    events = []
    for box in result.boxes:
        label = names[int(box.cls[0])]
        confidence = float(box.conf[0])
        if label in TARGET_CLASSES and confidence > THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            events.append((label, confidence, x1, y1, x2, y2))
    return events


def log_events(frame, events):
    file_stamp = time.strftime('%Y%m%d_%H%M%S')
    log_stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    cv2.imwrite(f'alerts/{file_stamp}.jpg', frame)
    with open('events.log', 'a') as f:
        for label, confidence, x1, y1, x2, y2 in events:
            f.write(f'{log_stamp},{label},{confidence:.2f},{x1},{y1},{x2},{y2}\n')
    print('Logged', len(events), 'event(s) at', log_stamp)


def draw_indicator(frame):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)
    cv2.putText(frame, 'LOGGING EVENT', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


def main():
    setup()
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Cannot open webcam')
        return

    last_log = 0
    indicator_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)[0]
        annotated = result.plot()
        events = get_events(result, model.names)
        now = time.time()

        if events and now - last_log >= COOLDOWN:
            log_events(annotated, events)
            last_log = now
            indicator_until = now + 1

        if now < indicator_until:
            draw_indicator(annotated)

        cv2.imshow('Surveillance', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
