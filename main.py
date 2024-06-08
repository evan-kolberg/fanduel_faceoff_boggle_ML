import cv2
import numpy as np
import glob
import os
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor

def load_images_and_masks(folder_path, resize_shape=(150, 150)):
    letter_images = {}
    masks = {}
    for letter_image_path in glob.glob(os.path.join(folder_path, "*.png")):
        letter = os.path.splitext(os.path.basename(letter_image_path))[0]
        img = cv2.imread(letter_image_path)
        resized_img = cv2.resize(img, resize_shape)
        letter_images[letter] = resized_img
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
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
    print(f"Overlay saved for cell ({row_idx}, {col_idx}) as {overlay_path}")

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

def split_and_process_image(image_path, letter_images, masks, color_ranges, overlay_folder, grid_size=4):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    cells = [[img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] for j in range(grid_size)] for i in range(grid_size)]
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
                color_row.append(''.join(color[:2].upper() for color in colors_present if colors_present[color] > 0))
            board.append(board_row)
            color_grid.append(color_row)
    return board, color_grid, cells

def main(image_path, letters_folder, output_file, overlay_folder):
    os.makedirs(overlay_folder, exist_ok=True)
    letter_images, masks = load_images_and_masks(letters_folder)
    color_ranges = {
        'red': [(201, 50, 40), (255, 210, 85)],
        'purple': [(125, 75, 235), (145, 100, 255)],
        'green': [(85, 150, 55), (100, 170, 75)],
        'yellow': [(200, 160, 0), (255, 230, 60)]
    }
    board, color_grid, cells = split_and_process_image(image_path, letter_images, masks, color_ranges, overlay_folder)
    with open(output_file, "w") as f:
        for row in board:
            f.write(" ".join(row) + "\n")
    print(f"Output saved to {output_file}")
    print("Board:")
    print(tabulate(board, tablefmt="grid"))
    print("\nColors detected:")
    print(tabulate(color_grid, tablefmt="grid"))

if __name__ == "__main__":
    image_path = "assets/boards/board6.png"
    letters_folder = "assets/letters"
    output_file = "output.txt"
    overlay_folder = "overlays"
    main(image_path, letters_folder, output_file, overlay_folder)


