import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def load_images_and_masks(folder_path):
    letter_images = {}
    masks = {}
    for letter_image_path in glob.glob(os.path.join(folder_path, "*.png")):
        letter = os.path.splitext(os.path.basename(letter_image_path))[0]
        img = cv2.imread(letter_image_path)
        letter_images[letter] = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        masks[letter] = mask
    return letter_images, masks

def process_image_cell(cell, masks, letter_images, color_ranges, row_idx, col_idx, overlay_folder):
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, mask_cell = cv2.threshold(gray_cell, 50, 255, cv2.THRESH_BINARY_INV)
    letter_scores = {letter: np.mean(cv2.absdiff(mask_cell, mask)) for letter, mask in masks.items()}
    best_match = min(letter_scores, key=letter_scores.get)
    overlay = cv2.addWeighted(cell, 0.4, letter_images[best_match], 0.6, 0)
    
    overlay_path = os.path.join(overlay_folder, f"overlay_{row_idx}_{col_idx}.png")
    cv2.imwrite(overlay_path, overlay)

    color_counts = {color: 0 for color in color_ranges}
    for row in cell[:20]:
        for pixel in row[:20]:
            for color, (lower_bound, upper_bound) in color_ranges.items():
                if (lower_bound[0] <= pixel[2] <= upper_bound[0] and
                    lower_bound[1] <= pixel[1] <= upper_bound[1] and
                    lower_bound[2] <= pixel[0] <= upper_bound[2]):
                    color_counts[color] += 1
                    return best_match, overlay_path, color_counts
    return best_match, overlay_path, color_counts

def split_and_process_image(image_path, letter_images, masks, color_ranges, overlay_folder, grid_size=4, target_size=(603, 603)):
    img = Image.open(image_path)
    img_resized = img.resize(target_size, Image.LANCZOS)
    image = np.array(img_resized)
    
    h, w, _ = image.shape
    print("Processing resized image with dimensions:", w, "x", h)

    h, w, _ = image.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    cells = [[image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] for j in range(grid_size)] for i in range(grid_size)]
    board = []
    color_grid = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_cell, cell, masks, letter_images, color_ranges, i, j, overlay_folder)
                   for i, row in enumerate(cells) for j, cell in enumerate(row)]
        processed_cells = [future.result() for future in futures]
        for i in range(0, len(processed_cells), grid_size):
            board_row = []
            color_row = []
            for j in range(grid_size):
                letter, overlay_path, colors_present = processed_cells[i + j]
                board_row.append(letter)
                color_row.append(''.join(color[:1].upper() for color in colors_present if colors_present[color] > 0))
            board.append(board_row)
            color_grid.append(color_row)
    return board, color_grid, cells
