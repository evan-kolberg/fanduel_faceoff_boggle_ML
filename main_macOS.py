import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import open_clip
import pyautogui
import torch
from PIL import Image, ImageGrab
from pynput import keyboard, mouse
from pynput.keyboard import Key, KeyCode
from scipy.interpolate import CubicSpline, interp1d
from sentence_transformers import util

from pyggle.lib.pyggle import Boggle, boggle


def imageEncoder(img: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(img).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img = model.encode_image(img)
    return img

def generateScore(img1: np.ndarray, img2: np.ndarray) -> float:
    img1, img2 = map(imageEncoder, (img1, img2))
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0]) * 100, 2)
    return score

def compareAllImages(img: np.ndarray, directory: str) -> List[Tuple[str, float]]:
    img1_tensor = imageEncoder(img)
    scores = []
    filenames = [f for f in os.listdir(directory) if f.endswith('.png')]
    def process_image(filename):
        image_path = os.path.join(directory, filename)
        img2 = cv2.imread(image_path)
        img2_tensor = imageEncoder(img2)
        cos_scores = util.pytorch_cos_sim(img1_tensor, img2_tensor)
        score = round(float(cos_scores[0][0]) * 100, 2)
        return filename, score
    with ThreadPoolExecutor() as executor:
        scores = list(executor.map(process_image, filenames))
    return scores

def capture_screen_region_for_colors(top_left: tuple, bottom_right: tuple) -> np.ndarray:
    # grab as RGB, convert once to BGR and return raw BGR
    img = ImageGrab.grab(bbox=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def capture_screen_region_for_comparison(top_left: tuple, bottom_right: tuple) -> np.ndarray:
    img = ImageGrab.grab(bbox=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def split_image_into_4x4_grid(image: np.ndarray) -> list[np.ndarray]:
    height, width = image.shape[:2]
    cell_width = width // 4
    cell_height = height // 4
    image_pieces = []
    for i in range(4):
        for j in range(4):
            start_y = i * cell_height
            start_x = j * cell_width
            end_y = (i + 1) * cell_height if i < 3 else height
            end_x = (j + 1) * cell_width if j < 3 else width
            piece = image[start_y:end_y, start_x:end_x]
            image_pieces.append(piece)
    return image_pieces

def get_most_similar_letter(scores: List[Tuple[str, float]]) -> Tuple[str, float]:
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0].replace('.png', ''), scores[0][1]

def binary_image_pieces(image_pieces: List[np.ndarray]) -> List[np.ndarray]:
    processed_pieces: List[np.ndarray] = []
    for piece in image_pieces:
        gray_piece = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_piece, 55, 255, cv2.THRESH_BINARY_INV)
        processed_pieces.append(mask)
    return processed_pieces

def save_images(image_pieces: List[np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i, piece in enumerate(image_pieces):
        output_path = os.path.join(output_dir, f"piece_{i}.png")
        cv2.imwrite(output_path, piece)

def list_to_board(lst: list) -> list[list[str]]:
    return [lst[i * 4: i * 4 + 4] for i in range(4)]

def get_average_color(image: np.ndarray, k: int = 3) -> np.ndarray:
    # simple per‑channel mean for consistent color readings
    flat = image.reshape(-1, 3).astype(np.float32)
    return flat.mean(axis=0)

def classify_and_store_bonus_tiles(color_pieces: List[np.ndarray]) -> Dict[str, List[Tuple[int, int]]]:
    bonus_tiles = {'DL': [], 'DW': [], 'TL': [], 'TW': []}
    for i, piece in enumerate(color_pieces):
        avg_color = get_average_color(piece)
        logging.debug(f"Piece {i} average color: {avg_color}")
        print(f"Piece {i} average color:", avg_color)
        x, y = i % 4, i // 4
        if np.allclose(avg_color, [93, 180, 203], atol=40):
            bonus_tiles['DL'].append((x, y))
        elif np.allclose(avg_color, [85, 180, 135], atol=40):
            bonus_tiles['DW'].append((x, y))
        elif np.allclose(avg_color, [70, 92, 168], atol=40):
            bonus_tiles['TL'].append((x, y))
        elif np.allclose(avg_color, [170, 90, 118], atol=40):
            bonus_tiles['TW'].append((x, y))
    return bonus_tiles

def calculate_word_score(word: str,
                         coords: Tuple[int, int],
                         board: List[List[str]],
                         bonus_tiles: Dict[str, int],
                         letter_points: Dict[str, int]) -> int:
    word_score = 0
    word_multipliers = []
    logging.debug(f'Scoring: {word}...')
    logging.debug(f'Coords: {coords}')
    logging.debug(f'Bonus tiles: {bonus_tiles}')
    for (x, y) in coords:
        letter = board[y][x]
        base_score = letter_points[letter]
        logging.debug(f'Letter: {letter}, Base Score: {base_score}')
        for bonus, positions in bonus_tiles.items():
            if (x, y) in positions:
                if bonus == 'TL':
                    base_score *= 3
                    logging.debug(f'Applied TL bonus at: {(x,y)}, new score: {base_score}')
                elif bonus == 'DL':
                    base_score *= 2
                    logging.debug(f'Applied DL bonus at: {(x,y)}, new score: {base_score}')
                elif bonus == 'TW':
                    word_multipliers.append(3)
                    logging.debug(f'Applied TW bonus at: {(x,y)}')
                elif bonus == 'DW':
                    word_multipliers.append(2)
                    logging.debug(f'Applied DW bonus at: {(x,y)}')
        word_score += base_score
    for multiplier in word_multipliers:
        word_score *= multiplier
        logging.debug(f'Applied word multiplier, new score: {word_score}')
    if len(word) >= 5:
        bonus_points = (len(word) - 4) * 5
        word_score += bonus_points
        logging.debug(f'Applied length bonus, new score: {word_score}')
    logging.debug(f'Final score for {word}: {word_score}')
    return word_score

def get_word_screen_coords(word: str, board_coords: list[tuple], top_left: tuple, bottom_right: tuple) -> list[tuple]:
    box_width = (bottom_right[0] - top_left[0]) // 4
    box_height = (bottom_right[1] - top_left[1]) // 4
    word_screen_coords = []
    for coord in board_coords:
        screen_x = coord[0] * box_width + box_width // 2 + top_left[0]
        screen_y = coord[1] * box_height + box_height // 2 + top_left[1]
        print(f"Screen coordinates for {coord}: ({screen_x}, {screen_y})")
        word_screen_coords.append((screen_x, screen_y))
    return word_screen_coords

def glide_mouse_to_positions(word_screen_coords: list[tuple], duration: float = 2.0, steps_multiplier_if_gliding: int = 3, glide: bool = True) -> None:
    if not glide:
        for coord in word_screen_coords:
            pyautogui.moveTo(coord[0], coord[1], duration)
            time.sleep(duration)
    elif len(word_screen_coords) >= 4:
        # Use Catmull-Rom spline
        x = [coord[0] for coord in word_screen_coords]
        y = [coord[1] for coord in word_screen_coords]
        t = np.arange(len(word_screen_coords))
        cs = CubicSpline(t, np.c_[x, y], bc_type='clamped')
        steps = steps_multiplier_if_gliding * len(word_screen_coords)
        for i in np.linspace(0, len(word_screen_coords) - 1, steps):
            pyautogui.moveTo(cs(i)[0], cs(i)[1], duration)
            time.sleep(duration / steps)
    else:
        # Use interpolation
        x = [coord[0] for coord in word_screen_coords]
        y = [coord[1] for coord in word_screen_coords]
        t = np.linspace(0, 1, len(word_screen_coords))
        fx = interp1d(t, x, kind='linear')
        fy = interp1d(t, y, kind='linear')
        steps = steps_multiplier_if_gliding * len(word_screen_coords)
        for i in np.linspace(0, 1, steps):
            pyautogui.moveTo(fx(i), fy(i), duration)
            time.sleep(duration / steps)

def on_press(key: Union[Key, KeyCode]) -> None: # callback function
    if key == Key.enter:
        print('Enter key pressed. Current mouse position is:', mouse_controller.position)
        mouse_positions.append(mouse_controller.position)
    if key == Key.shift:
        print('Shift key pressed. Exiting...')
        exit()

def get_words_until_min_letters(word_scores: List[Tuple[str, int]], min_letters: int) -> List[Tuple[str, int]]:
    total_letters = 0
    for i, (word, _) in enumerate(word_scores):
        total_letters += len(word)
        if total_letters >= min_letters:
            return word_scores[:i+1]
    return word_scores
        

if __name__ == '__main__':
    logging.basicConfig(filename='scoring_debug.log', level=logging.DEBUG)

    letter_points = {
        'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
        'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
        'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
    }

    
    device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
    print('Using device:', device)
    print('Loading model...')
    # use a lighter ViT‑B‑32 for faster encoding
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.to(device)
    print('Model loaded.')

    mouse_controller = mouse.Controller()
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    while True:
        mouse_positions = []
        print('Listening for green‑check, then board top‑left and bottom‑right clicks...')
        # need 3 clicks: green‑check button, board top-left, board bottom-right
        while len(mouse_positions) < 3: pass

        green_check  = (round(mouse_positions[0][0]), round(mouse_positions[0][1]))
        top_left     = (round(mouse_positions[1][0]), round(mouse_positions[1][1]))
        bottom_right = (round(mouse_positions[2][0]), round(mouse_positions[2][1]))

        print('Green check position:',           mouse_positions[0])
        print('Board top-left corner position:',  mouse_positions[1])
        print('Board bottom-right corner position:', mouse_positions[2])
        print('Positions captured.')

        region = capture_screen_region_for_comparison(top_left, bottom_right)

        image_pieces = split_image_into_4x4_grid(region)
        image_pieces = binary_image_pieces(image_pieces)
        save_images(image_pieces, "pieces_output")

        color_region = capture_screen_region_for_colors(top_left, bottom_right)
        color_pieces = split_image_into_4x4_grid(color_region)
        save_images(color_pieces, "color_pieces_output")
        bonus_tiles = classify_and_store_bonus_tiles(color_pieces)

        letters = []
        for index, piece in enumerate(image_pieces):
            scores = compareAllImages(piece, "control_group")
            most_similar_letter, highest_score = get_most_similar_letter(scores)
            letters.append(most_similar_letter)
            logging.debug(f'Piece {index} - Most similar letter: {most_similar_letter} with score: {highest_score}')
            print(f'Piece {index} - Most similar letter:', most_similar_letter, 'with score:', highest_score)

        board = list_to_board(letters)
        print('Board:', board)
        print('Bonus tiles:', bonus_tiles)
        logging.debug(f'Board: {board}')
        logging.debug(f'Bonus tiles: {bonus_tiles}')

        boggle = Boggle(board)
        solved = boggle.solve()
        print(solved)

        word_scores = []
        for word, coords in solved.items():
            score = calculate_word_score(word, coords, board, bonus_tiles, letter_points)
            word_scores.append((word, score))

        word_scores.sort(key=lambda x: x[1], reverse=False)

        for word, score in word_scores:
            print(f"{word}: {score}")
        
        word_scores.sort(key=lambda x: x[1], reverse=True)

        # *** minumum letters to enter, rounds up a word *** 
        filtered_entries = get_words_until_min_letters(word_scores, 135) # TWEAK
        print("Filtered entries:", filtered_entries)

        # move slowly to first pos of first word
        if filtered_entries:
            first_word, _ = filtered_entries[0]
            first_board_coords = solved[first_word]
            first_word_screen_coords = get_word_screen_coords(first_word, first_board_coords, top_left, bottom_right)
            glide_mouse_to_positions([mouse_controller.position, first_word_screen_coords[0]], duration=0.08, glide=False)

        for i in range(len(filtered_entries)):
            word, score = filtered_entries[i]
            print(f"Entering word: {word} with score: {score}")
            board_coords = solved[word]
            word_screen_coords = get_word_screen_coords(word, board_coords, top_left, bottom_right)

            # click near center of each tile with small randomness
            cell_w = (bottom_right[0] - top_left[0]) / 4
            cell_h = (bottom_right[1] - top_left[1]) / 4
            for grid_x, grid_y in board_coords:
                center_x = top_left[0] + grid_x * cell_w + cell_w / 2
                center_y = top_left[1] + grid_y * cell_h + cell_h / 2
                rand_x = center_x + random.uniform(-cell_w * 0.25, cell_w * 0.25)
                rand_y = center_y + random.uniform(-cell_h * 0.25, cell_h * 0.25)
                pyautogui.click(rand_x, rand_y)
                time.sleep(random.uniform(0.01, 0.2))
            pyautogui.click(green_check[0], green_check[1])
            time.sleep(random.uniform(0.01, 0.4))

            # If there is a next word, slowly move to its first position
            if i < len(filtered_entries) - 1:
                next_word, _ = filtered_entries[i + 1]
                next_board_coords = solved[next_word]
                next_word_screen_coords = get_word_screen_coords(next_word, next_board_coords, top_left, bottom_right)
                glide_mouse_to_positions([mouse_controller.position, next_word_screen_coords[0]], duration=0.08, glide=False) # TWEAK





















