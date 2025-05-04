import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
from pynput import keyboard
from pynput.keyboard import Key
import os

def get_average_color(image: np.ndarray, k: int = 3) -> np.ndarray:
    # simple per-channel mean to ensure consistency
    flat = image.reshape(-1, 3).astype(np.float32)
    avg_color = flat.mean(axis=0)
    return avg_color

def wait_for_enter() -> tuple:
    coords = {}
    def on_press(key):
        if key == Key.enter:
            coords['pos'] = pyautogui.position()
            return False
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    return coords['pos']

def main():
    print("Color Calibrating Tool (global ENTER capture)")
    # prepare directory for tile image captures
    save_dir = "captured_tiles"
    os.makedirs(save_dir, exist_ok=True)
    counter = 1

    while True:
        print("\nMove mouse to top-left of tile, then press ENTER.")
        tl = wait_for_enter()
        print("Top-left recorded at", tl)

        print("Move mouse to bottom-right of tile, then press ENTER.")
        br = wait_for_enter()
        print("Bottom-right recorded at", br)

        # grab and compute average
        img = ImageGrab.grab(bbox=(tl.x, tl.y, br.x, br.y))
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        avg = get_average_color(arr)
        print(f"Average BGR color: {avg.astype(int)}")

        # save captured tile image
        file_path = os.path.join(save_dir, f"tile_{counter}.png")
        cv2.imwrite(file_path, arr)
        print(f"Saved tile image to {file_path}")
        counter += 1

        print("Ready for next tile capture...")

if __name__ == "__main__":
    main()

