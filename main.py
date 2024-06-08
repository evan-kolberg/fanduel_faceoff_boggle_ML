import cv2
import numpy as np
import glob
import os
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor

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
            row.append(cell)
        cells.append(row)
    
    return cells

def load_letter_images(folder_path):
    letter_images = {}
    for letter_image_path in glob.glob(folder_path + "/*.png"):
        filename = os.path.basename(letter_image_path)
        letter = filename.split(".")[0]
        img = cv2.imread(letter_image_path)
        if img is None:
            print(f"Error: Could not read letter image at {letter_image_path}")
            continue
    
        resized_img = cv2.resize(img, (150, 150))
        letter_images[letter] = resized_img
    return letter_images

def filter_black_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return mask

def precompute_masks(letter_images):
    masks = {}
    for letter, letter_img in letter_images.items():
        masks[letter] = filter_black_text(letter_img)
    return masks

def match_letter(cell, masks, row_idx, col_idx):
    mask_cell = filter_black_text(cell)
    match_scores = {}
    
    for letter, mask_letter in masks.items():
        error = calculate_mse(mask_cell, mask_letter)
        print(f"MSE ({row_idx}, {col_idx}) | {letter}: {error}")
        match_scores[letter] = error
        
    return match_scores

def calculate_mse(img1, img2):
    diff = cv2.absdiff(img1, img2)
    err = np.mean(diff)
    return err

def overlay_images(cell, best_match_img):
    alpha = 0.4  
    overlay = cv2.addWeighted(cell, alpha, best_match_img, 1 - alpha, 0)
    return overlay

def process_cell(row_idx, col_idx, cell, masks, letter_images):
    letter_scores = match_letter(cell, masks, row_idx, col_idx)
    best_match = min(letter_scores, key=letter_scores.get)
    overlay = overlay_images(cell, letter_images[best_match])
    overlay_path = f"overlays/overlay_{row_idx}_{col_idx}.png"
    cv2.imwrite(overlay_path, overlay)
    print(f"Overlay saved for cell ({row_idx}, {col_idx}) as {overlay_path}")
    return best_match, overlay_path, (row_idx, col_idx)

def main(image_path, letters_folder):
    cells = split_image(image_path)
    if cells is None:
        return
    letter_images = load_letter_images(letters_folder)
    if not letter_images:
        print(f"Error: No letter images loaded from {letters_folder}")
        return
    
    masks = precompute_masks(letter_images)

    board = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for row_idx, row in enumerate(cells):
            row_data = []
            for col_idx, cell in enumerate(row):
                futures.append(executor.submit(process_cell, row_idx, col_idx, cell, masks, letter_images))
        
        processed_cells = [future.result() for future in futures]
        for i in range(0, len(processed_cells), 4):
            row_data = [processed_cells[i+j][0] for j in range(4)]
            board.append(row_data)

    print(tabulate(board, tablefmt="grid"))

if __name__ == "__main__":
    image_path = "assets/boards/board3.png"
    letters_folder = "assets/letters"
    main(image_path, letters_folder)


