import cv2
import numpy as np
from PIL import ImageGrab
import pygetwindow as gw
from ultralytics import YOLO

TARGET_WINDOW_TITLE = "BlueStacks App Player"
CAPTURE_WIDTH = 544 # do not touch 
CAPTURE_HEIGHT = 928 # do not touch

model = YOLO('')

def capture_window(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        if window:
            center_x = window.left + window.width // 2
            center_y = window.top + window.height // 2

            left = max(center_x - CAPTURE_WIDTH // 2, 0)
            top = max(center_y - CAPTURE_HEIGHT // 2, 0)
            right = left + CAPTURE_WIDTH
            bottom = top + CAPTURE_HEIGHT

            img = ImageGrab.grab(bbox=(left, top, right, bottom))
            img_np = np.array(img)
            frame_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            return frame_gray
    except IndexError:
        print(f"No window found with title: {window_title}")
    return None

def run_inference(model, image):
    results = model(image)
    return results

def main():
    while True:
        screen = capture_window(TARGET_WINDOW_TITLE)
        if screen is not None:
            results = run_inference(model, screen)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for box, score, class_id in zip(boxes, scores, class_ids):
                    if score > 0.5:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{class_id}: {int(score * 100)}%'
                        cv2.putText(screen, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Object Detection', screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


