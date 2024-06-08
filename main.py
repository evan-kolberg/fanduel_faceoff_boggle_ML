import cv2
import numpy as np
import glob
import os
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor

def split_image(image_path, grid_size=4):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    h, w, _ = img.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    cells = [[img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] for j in range(grid_size)] for i in range(grid_size)]
    return cells

def load_letter_images(folder_path, resize_shape=(150, 150)):
    letter_images = {}
    for letter_image_path in glob.glob(os.path.join(folder_path, "*.png")):
        letter = os.path.splitext(os.path.basename(letter_image_path))[0]
        img = cv2.imread(letter_image_path)
        if img is None:
            print(f"Error: Could not read letter image at {letter_image_path}")
            continue
        resized_img = cv2.resize(img, resize_shape)
        letter_images[letter] = resized_img
    return letter_images

def filter_black_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    return mask

def precompute_masks(letter_images):
    return {letter: filter_black_text(img) for letter, img in letter_images.items()}

def match_letter(cell, masks):
    mask_cell = filter_black_text(cell)
    return {letter: calculate_mse(mask_cell, mask) for letter, mask in masks.items()}

def calculate_mse(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return np.mean(diff)

def overlay_images(cell, best_match_img, alpha=0.4):
    return cv2.addWeighted(cell, alpha, best_match_img, 1 - alpha, 0)

def check_colors_in_cell(cell, color_ranges):
    color_counts = {color: 0 for color in color_ranges}
    for row in cell[:20]:
        for pixel in row[:20]:
            for color, (lower_bound, upper_bound) in color_ranges.items():
                if (lower_bound[0] <= pixel[2] <= upper_bound[0] and
                    lower_bound[1] <= pixel[1] <= upper_bound[1] and
                    lower_bound[2] <= pixel[0] <= upper_bound[2]):
                    color_counts[color] += 1
                    return color_counts
    return color_counts

def process_cell(row_idx, col_idx, cell, masks, letter_images, color_ranges):
    letter_scores = match_letter(cell, masks)
    colors_present = check_colors_in_cell(cell, color_ranges)
    best_match = min(letter_scores, key=letter_scores.get)
    overlay = overlay_images(cell, letter_images[best_match])
    overlay_path = f"overlays/overlay_{row_idx}_{col_idx}.png"
    cv2.imwrite(overlay_path, overlay)
    print(f"Overlay saved for cell ({row_idx}, {col_idx}) as {overlay_path}")
    detected_colors = [f"{color}: {count}" for color, count in colors_present.items() if count > 0]
    print(f"Colors detected in cell ({row_idx}, {col_idx}): {', '.join(detected_colors)}")
    return best_match, overlay_path, colors_present, (row_idx, col_idx)

def main(image_path, letters_folder, output_file):
    cells = split_image(image_path)
    if cells is None:
        return
    letter_images = load_letter_images(letters_folder)
    if not letter_images:
        print(f"Error: No letter images loaded from {letters_folder}")
        return
    masks = precompute_masks(letter_images)
    color_ranges = {
        'red': [(201, 50, 40), (255, 210, 85)],
        'purple': [(125, 75, 235), (145, 100, 255)],
        'green': [(85, 150, 55), (100, 170, 75)],
        'yellow': [(200, 160, 0), (255, 230, 60)]
    }
    board = []
    color_grid = []
    output_data = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_cell, row_idx, col_idx, cell, masks, letter_images, color_ranges)
                   for row_idx, row in enumerate(cells) for col_idx, cell in enumerate(row)]
        processed_cells = [future.result() for future in futures]
        for i in range(0, len(processed_cells), 4):
            board_row_data = []
            color_row_data = []
            for j in range(4):
                letter, overlay_path, colors_present, (row_idx, col_idx) = processed_cells[i+j]
                board_row_data.append(letter)
                output_data.append(letter)
                if any(colors_present[color] > 0 for color in color_ranges):
                    ticker = ''.join(color[:2].upper() for color in color_ranges if colors_present[color] > 0)
                    color_row_data.append(ticker)
                else:
                    color_row_data.append("")
            board.append(board_row_data)
            color_grid.append(color_row_data)
    print("Board:")
    print(tabulate(board, tablefmt="grid"))
    print("\nColors detected:")
    print(tabulate(color_grid, tablefmt="grid"))
    with open(output_file, "w") as f:
        for i in range(0, len(output_data), 4):
            line = " ".join(output_data[i:i+4]) + "\n"
            f.write(line)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    image_path = "assets/boards/board6.png"
    letters_folder = "assets/letters"
    output_file = "output.txt"
    main(image_path, letters_folder, output_file)




