import os
import cv2
import math
import time
import torch
import open_clip
import pyautogui
import numpy as np
from PIL import Image, ImageGrab
from pynput import mouse, keyboard
from sentence_transformers import util
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Controller, Button
from typing import List, Tuple, Union, Dict
from pyggle.lib.pyggle import Boggle, boggle
from scipy.interpolate import CubicSpline, interp1d

def imageEncoder(img: np.ndarray) -> torch.Tensor:
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def generateScore(img1: np.ndarray, img2: np.ndarray) -> float:
    img1, img2 = map(imageEncoder, (img1, img2))
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

def compareAllImages(img: np.ndarray, directory: str) -> List[Tuple[str, float]]:
    scores = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            img2 = cv2.imread(image_path)
            score = generateScore(img, img2)
            scores.append((filename, score))
    return scores

def capture_screen_region_for_colors(top_left: tuple, bottom_right: tuple) -> np.ndarray:
    img = ImageGrab.grab(bbox=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    color_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    return np.array(color_image)

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
    processed_pieces = []
    lower_black = np.array([0, 0, 0], dtype = "uint8")
    upper_black = np.array([50, 50, 50], dtype = "uint8")
    for piece in image_pieces:
        mask = cv2.inRange(piece, lower_black, upper_black)
        processed_pieces.append(mask)
    return processed_pieces

def save_images(image_pieces: List[np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for i, piece in enumerate(image_pieces):
        output_path = os.path.join(output_dir, f"piece_{i}.png")
        cv2.imwrite(output_path, piece)

def list_to_board(lst: list) -> list[list: str]:
    return [lst[i*4 : i*4 + 4] for i in range(4)]

def get_average_color(image: np.ndarray, k: int = 3) -> np.ndarray:
    pixels = image.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(pixels, k, None, 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    average_color = centers[np.argmax(np.bincount(labels.flatten()))]
    return average_color

def classify_and_store_bonus_tiles(color_pieces: List[np.ndarray]) -> Dict[str, List[Tuple[int, int]]]:
    bonus_tiles = {'DL': [], 'DW': [], 'TL': [], 'TW': []}
    for i, piece in enumerate(color_pieces):
        avg_color = get_average_color(piece)
        print(f"Piece {i} average color:", avg_color)
        x, y = i % 4, i // 4
        if np.allclose(avg_color, [230, 190, 55], atol=20):
            bonus_tiles['DL'].append((x, y))
        elif np.allclose(avg_color, [100, 180, 55], atol=20):
            bonus_tiles['DW'].append((x, y))
        elif np.allclose(avg_color, [245, 100, 55], atol=20):
            bonus_tiles['TL'].append((x, y))
        elif np.allclose(avg_color, [145, 100, 230], atol=20):
            bonus_tiles['TW'].append((x, y))
    return bonus_tiles

def calculate_word_score(word: str, 
                         coords: Tuple[int, int], 
                         board: List[List[str]], 
                         bonus_tiles: Dict[str, int], 
                         letter_points: Dict[str, int]) -> int:
    word_score = 0
    word_multipliers = []
    for (x, y) in coords:
        letter = board[y][x]
        base_score = letter_points[letter]
        for bonus, positions in bonus_tiles.items():
            if (x, y) in positions:
                if bonus == 'TL':
                    base_score *= 3
                elif bonus == 'DL':
                    base_score *= 2
                elif bonus == 'TW':
                    word_multipliers.append(3)
                elif bonus == 'DW':
                    word_multipliers.append(2)
        word_score += base_score
    for multiplier in word_multipliers:
        word_score *= multiplier
    if len(word) >= 5:
        bonus_points = (len(word) - 4) * 5
        word_score += bonus_points
    return word_score

def get_word_screen_coords(word: str, board_coords: list[tuple], top_left: tuple, bottom_right: tuple) -> list[tuple]:
    box_width = (bottom_right[0] - top_left[0]) // 4
    box_height = (bottom_right[1] - top_left[1]) // 4
    print(f"Box width: {box_width}, Box height: {box_height}")
    word_screen_coords = []
    print(board_coords)
    for coord in board_coords:
        screen_x = coord[0] * box_width + box_width // 2 + top_left[0]
        screen_y = coord[1] * box_height + box_height // 2 + top_left[1]
        print(f"Screen coordinates for {coord}: ({screen_x}, {screen_y})")
        word_screen_coords.append((screen_x, screen_y))
    return word_screen_coords

def glide_mouse_to_positions(word_screen_coords: list[tuple], duration: float = 2.0):
    if len(word_screen_coords) >= 4:
        # Use Catmull-Rom spline
        x = [coord[0] for coord in word_screen_coords]
        y = [coord[1] for coord in word_screen_coords]
        t = np.arange(len(word_screen_coords))
        cs = CubicSpline(t, np.c_[x, y], bc_type='clamped')
        steps = int(duration * 100)
        for i in np.linspace(0, len(word_screen_coords) - 1, steps):
            pyautogui.moveTo(cs(i)[0], cs(i)[1], duration / steps)
            time.sleep(duration / steps)
    else:
        # Use interpolation
        x = [coord[0] for coord in word_screen_coords]
        y = [coord[1] for coord in word_screen_coords]
        t = np.linspace(0, 1, len(word_screen_coords))
        fx = interp1d(t, x, kind='linear')
        fy = interp1d(t, y, kind='linear')
        steps = int(duration * 1000)
        for i in np.linspace(0, 1, steps):
            pyautogui.moveTo(fx(i), fy(i), duration / steps)
            time.sleep(duration / steps)

def on_press(key: Union[Key, KeyCode]) -> None: # callback function
    if key == Key.enter:
        print('Enter key pressed. Current mouse position is:', mouse_controller.position)
        mouse_positions.append(mouse_controller.position)

if __name__ == '__main__':

    letter_points = {
        'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
        'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
        'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
    }

    print('Loading model...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device:', torch.cuda.get_device_name(0))
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained='laion400m_e32')
    model.to(device)
    print('Model loaded.')

    while True:
        mouse_positions = []
        mouse_controller = mouse.Controller()
        keyboard_listener = keyboard.Listener(on_press=on_press)
        keyboard_listener.start()
        print('Listening...')

        while len(mouse_positions) < 2: pass

        top_left = mouse_positions[0]
        bottom_right = mouse_positions[1]
        print('First mouse position:', mouse_positions[0])
        print('Second mouse position:', mouse_positions[1])
        keyboard_listener.stop()
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
            print('Most similar letter:', most_similar_letter, 'with score:', highest_score)

        board = list_to_board(letters)
        print('Board:', board)
        print('Bonus tiles:', bonus_tiles)

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

        mouse_controller = Controller()
        for word, score in word_scores[:12]: # num of words to enter before next round
            print(f"Entering word: {word} with score: {score}")
            board_coords = solved[word]
            word_screen_coords = get_word_screen_coords(word, board_coords, top_left, bottom_right)
            print(f"Word screen coordinates: {word_screen_coords}")
            
            mouse_controller.position = word_screen_coords[0]
            mouse_controller.press(Button.left)
            glide_mouse_to_positions(word_screen_coords, duration=0.15)
            mouse_controller.release(Button.left)
            time.sleep(1)












