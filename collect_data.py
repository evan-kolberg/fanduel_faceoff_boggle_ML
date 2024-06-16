import cv2
import numpy as np
from PIL import ImageGrab
import pygetwindow as gw
from pynput import keyboard
import os
import datetime

TARGET_WINDOW_TITLE = "BlueStacks App Player"
SCREENSHOTS_FOLDER = "data"
CAPTURE_SIZE = 1024 # 1:1 aspect ratio

def capture_window(window_title):
    try:
        window = gw.getWindowsWithTitle(window_title)[0]
        if window:
            center_x = window.left + window.width // 2
            center_y = window.top + window.height // 2

            left = max(center_x - CAPTURE_SIZE // 2, 0)
            top = max(center_y - CAPTURE_SIZE // 2, 0)
            right = left + CAPTURE_SIZE
            bottom = top + CAPTURE_SIZE

            img = ImageGrab.grab(bbox=(left, top, right, bottom))
            img_np = np.array(img)
            frame_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            return frame_gray
    except IndexError:
        print(f"No window found with title: {window_title}")
    return None

def save_screenshot(frame, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"screenshot_{timestamp}.png")

    cv2.imwrite(filename, frame)
    print(f"Screenshot saved to {filename}")

def on_press(key):
    try:
        if key.char == 's':
            frame = capture_window(TARGET_WINDOW_TITLE)
            if frame is not None:
                save_screenshot(frame, SCREENSHOTS_FOLDER)
    except AttributeError:
        pass

def main():
    print("Press 's' to capture the window")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()


