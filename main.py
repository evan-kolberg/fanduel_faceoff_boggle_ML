import cv2
import numpy as np
import glob
import os
from tabulate import tabulate
from PIL import Image

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
        
            pil_img = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        
            temp_file_path = f"temp_cell_{i}_{j}.png"
            pil_img.save(temp_file_path)
            row.append(temp_file_path)
        cells.append(row)
    
    return cells

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
    
        resized_img = cv2.resize(img, (150, 150))
        letter_images[letter] = resized_img
        lookup_table[filename.lower()] = letter
    return letter_images, lookup_table

def filter_black_text(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to get a binary mask of black regions
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return mask

def match_letter(cell, letter_images, cell_coords):
    mask_cell = filter_black_text(cell)
    match_scores = {}
    
    for letter, letter_img in letter_images.items():
        mask_letter = filter_black_text(letter_img)
        error = calculate_mse(mask_cell, mask_letter)
        match_scores[letter] = error
        print(f"Deviation {cell_coords}: {letter}: {error}")
        
    return match_scores

def calculate_mse(img1, img2):
    diff = cv2.absdiff(img1, img2)
    err = np.mean(diff)
    return err

def overlay_images(cell, best_match_img):
    alpha = 0.5  # transparency factor
    overlay = cv2.addWeighted(cell, alpha, best_match_img, 1 - alpha, 0)
    return overlay

def main(image_path, letters_folder):
    cells = split_image(image_path)
    if cells is None:
        return
    letter_images, lookup_table = load_letter_images(letters_folder)
    if not letter_images:
        print(f"Error: No letter images loaded from {letters_folder}")
        return

    board = []

    for row_idx, row in enumerate(cells):
        row_data = []
        for col_idx, cell_path in enumerate(row):
            cell = cv2.imread(cell_path)
            os.remove(cell_path) 
            
        
            letter_scores = match_letter(cell, letter_images, (row_idx, col_idx))
            best_match = min(letter_scores, key=letter_scores.get)
            
            row_data.append(best_match)
            
            # Visualization
            best_match_img = letter_images[best_match]
            overlay = overlay_images(cell, best_match_img)
            overlay_path = f"overlays/overlay_{row_idx}_{col_idx}.png"
            cv2.imwrite(overlay_path, overlay)
            print(f"Overlay saved for cell ({row_idx}, {col_idx}) as {overlay_path}")
        
        board.append(row_data)
    
    print(tabulate(board, tablefmt="grid"))

if __name__ == "__main__":
    image_path = "assets/boards/board2.png"
    letters_folder = "assets/letters"
    main(image_path, letters_folder)
