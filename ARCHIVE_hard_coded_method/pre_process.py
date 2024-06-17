import os
import cv2
import numpy as np

def extract_black_text(input_image_path, output_dir):
    # Read image
    input_image = cv2.imread(input_image_path)

    # Conversion to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Threshold image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply blur
    blur = cv2.blur(binary_image, (10, 10))

    # Save the blurred image
    output_blurred_path = os.path.join(output_dir, "blurred_" + os.path.basename(input_image_path))
    cv2.imwrite(output_blurred_path, blur)

if __name__ == "__main__":
    input_directory = "hard-coded-method/assets/letters"
    output_directory = "hard-coded-method/output"
    
    # Process all images in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_image_path = os.path.join(input_directory, filename)
            extract_black_text(input_image_path, output_directory)
