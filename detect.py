import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

model = YOLO('runs/detect/boggle-model-8n/weights/best.pt')

def capture_screen(bbox=(0, 0, 1024, 1024)):
    screen = np.array(pyautogui.screenshot(region=bbox))
    return screen

def run_inference(model, image):
    results = model(image)
    return results

def main():
    while True:
        screen = capture_screen()

        results = run_inference(model, screen)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, score, class_id in zip(boxes, scores, class_ids):
                if score > 0.7:
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


