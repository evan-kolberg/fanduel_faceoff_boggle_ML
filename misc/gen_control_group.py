import os
import cv2
import numpy as np
from typing import List

def extract_black_pixels_from_images(image_pieces: List[np.ndarray]) -> List[np.ndarray]:
    processed_pieces: List[np.ndarray] = []
    lower_black: np.ndarray = np.array([0, 0, 0], dtype = "uint8")
    upper_black: np.ndarray = np.array([48, 48, 48], dtype = "uint8")
    for piece in image_pieces:
        mask: np.ndarray = cv2.inRange(piece, lower_black, upper_black)
        processed_pieces.append(mask)
    return processed_pieces

def save(input_image_path: str, output_dir: str) -> None:
    input_image: np.ndarray = cv2.imread(input_image_path)
    binary_image: np.ndarray = extract_black_pixels_from_images([input_image])[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path: str = os.path.join(output_dir, os.path.basename(input_image_path))
    cv2.imwrite(output_path, binary_image)

if __name__ == "__main__":
    input_directory: str = "raw_basic_letters"
    output_directory: str = "control_group"
    for filename in os.listdir(input_directory):
        if filename.lower().endswith((".png")):
            input_image_path: str = os.path.join(input_directory, filename)
            save(input_image_path, output_directory)






