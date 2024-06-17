import os
import cv2

def extract_black_text(input_image_path, output_dir):
    input_image = cv2.imread(input_image_path)

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    output_path = os.path.join(output_dir, os.path.basename(input_image_path))
    cv2.imwrite(output_path, binary_image)



if __name__ == "__main__":
    input_directory = "raw_basic_letters"
    output_directory = "binary_letters"
    
    for filename in os.listdir(input_directory):
        if filename.lower().endswith((".png")):
            input_image_path = os.path.join(input_directory, filename)
            extract_black_text(input_image_path, output_directory)




