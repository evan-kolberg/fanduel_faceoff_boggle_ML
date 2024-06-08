import pyautogui
import cv2
import numpy as np
import os
from PIL import Image

def capture_screen_region(region):
    screenshot = pyautogui.screenshot(region=region)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    return screenshot

def select_capture_region():
    input("\nTop Left. Press Enter to capture point... ")
    x1, y1 = pyautogui.position()
    print(f"Top-left point selected: ({x1}, {y1})")

    input("\nBottom Right. Press Enter to capture point... ")
    x2, y2 = pyautogui.position()
    print(f"Bottom-right point selected: ({x2}, {y2})")

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    width = x2 - x1
    height = y2 - y1

    print(f"Selected region dimensions: Width: {width}, Height: {height}")
    return (x1, y1, width, height)

def save_captured_image(img, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    cv2.imwrite(file_path, img)
    print(f"Captured image saved to {file_path}")
    return file_path

