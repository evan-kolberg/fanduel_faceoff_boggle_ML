import os
from tabulate import tabulate
from image_processing import load_images_and_masks, split_and_process_image
from screen_capture import select_capture_region, capture_screen_region, save_captured_image

def main(letters_folder, output_file, overlay_folder):
    os.makedirs(overlay_folder, exist_ok=True)
    letter_images, masks = load_images_and_masks(letters_folder)
    color_ranges = {
        'red': [(210, 50, 40), (255, 210, 85)],
        'purple': [(125, 75, 235), (145, 100, 255)],
        'green': [(85, 150, 55), (100, 170, 75)],
        'yellow': [(230, 200, 0), (255, 255, 100)]
    }
    
    region = select_capture_region()
    img = capture_screen_region(region)
    
    captured_image_path = "capture/captured_board.png"
    save_captured_image(img, "capture", "captured_board.png")
    
    board, color_grid, cells = split_and_process_image(captured_image_path, letter_images, masks, color_ranges, overlay_folder)
    
    with open(output_file, "w") as f:
        for row in board:
            f.write(" ".join(row) + "\n")
    
    print(tabulate(board, tablefmt="grid"))
    print(tabulate(color_grid, tablefmt="grid"))

if __name__ == "__main__":
    letters_folder = "assets/letters"
    output_file = "output.txt"
    overlay_folder = "overlays"
    main(letters_folder, output_file, overlay_folder)

