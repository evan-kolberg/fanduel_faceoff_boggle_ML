import cv2
import numpy as np
import glob
import os
from PIL import Image

# Function to split the input image into a 4x4 grid and save temporary files
def split_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    h, w, _ = img.shape
    grid_size = 4
    cell_h, cell_w = h // grid_size, w // grid_size
    cells = []

    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # Convert cell to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            # Save cell as temporary image file
            temp_file_path = f"temp_cell_{i}_{j}.png"
            pil_img.save(temp_file_path)
            row.append(temp_file_path)
        cells.append(row)
    
    return cells

# Function to load letter images and generate lookup table
def load_letter_images(folder_path):
    letter_images = {}
    lookup_table = {}
    for letter_image_path in glob.glob(folder_path + "/*.png"):
        filename = os.path.basename(letter_image_path)
        letter = filename.split(".")[0]
        img = cv2.imread(letter_image_path)
        if img is None:
            print(f"Error: Could not read letter image at {letter_image_path}")
            continue
        # Resize the image to a fixed size for consistency
        resized_img = cv2.resize(img, (150, 150))
        letter_images[letter] = resized_img
        lookup_table[filename.lower()] = letter
    return letter_images, lookup_table

# Function to calculate mean squared error
def calculate_mse(img1, img2):
    diff = cv2.absdiff(img1, img2)
    err = np.mean(diff)
    return err

# Function to find the best matching letter
def match_letter(cell, letter_images):
    cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    match_scores = {}
    
    for letter, letter_img in letter_images.items():
        # Convert letter image to grayscale
        letter_img_gray = cv2.cvtColor(letter_img, cv2.COLOR_BGR2GRAY)
        error = calculate_mse(cell_gray, letter_img_gray)
        match_scores[letter] = error
        
    return match_scores

# Function to identify special condition
def identify_special_condition(cell):
    # Define color thresholds for special conditions
    color_thresholds = {
        'none': [(160, 190), (130, 160), (100, 130)],  # No special
        'dl': [(180, 255), (150, 180), (0, 50)],       # Yellow
        'dw': [(0, 80), (100, 180), (0, 80)],          # Green
        'tl': [(0, 80), (0, 80), (100, 255)],          # Red
        'tw': [(130, 200), (50, 130), (200, 255)]      # Purple
    }
    
    cell_hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(cell_hsv)
    
    color_means = [np.mean(channel) for channel in (h, s, v)]
    
    special_scores = {}
    
    for special, thresholds in color_thresholds.items():
        distances = [abs(mean - threshold) for mean, threshold in zip(color_means, thresholds)]
        distance = sum(distances)
        special_scores[special] = distance
    
    return special_scores

def main(image_path, letters_folder):
    cells = split_image(image_path)
    if cells is None:
        return
    letter_images, lookup_table = load_letter_images(letters_folder)
    if not letter_images:
        print(f"Error: No letter images loaded from {letters_folder}")
        return

    board = ""

    for row_idx, row in enumerate(cells):
        for col_idx, cell_path in enumerate(row):
            cell = cv2.imread(cell_path)
            os.remove(cell_path)  # Delete temporary file after reading
            
            # Match letter and calculate closeness scores
            letter_scores = match_letter(cell, letter_images)
            # Identify special condition and calculate color scores
            special_scores = identify_special_condition(cell)
            
            best_match = min(letter_scores, key=letter_scores.get)
            board += best_match
            
            # Print deviation scores for each letter
            print(f"Deviation for  {row_idx}, {col_idx}:")
            for letter, score in letter_scores.items():
                print(f"  - {letter.upper()}: {score}")
            print("Special condition color scores:")
            for special, score in special_scores.items():
                print(f"  - {special}: {score}")
        board += "\n"
    
    print(board)
    with open("output.txt", "w") as file:
        file.write(board)

if __name__ == "__main__":
    image_path = "assets/boards/Screenshot 2024-06-07 171426.png"
    letters_folder = "assets/letters"
    print(f"Reading from image path: {image_path}")
    print(f"Reading from letters folder: {letters_folder}")
    main(image_path, letters_folder)