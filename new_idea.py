import os
import cv2
import torch
import open_clip
import numpy as np
from PIL import Image, ImageGrab
from pynput import mouse, keyboard
from typing import List, Tuple, Union
from sentence_transformers import util
from pynput.keyboard import Key, KeyCode
from pyggle.lib.pyggle import Boggle, boggle


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
    hsv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
    return np.array(hsv_image)

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

def get_most_similar_letter(scores: List[Tuple[str, float]]) -> str:
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0].replace('.png', '')

def binary_image_pieces(image_pieces: List[np.ndarray]) -> List[np.ndarray]:
    processed_pieces = []
    for piece in image_pieces:
        gray_image = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_pieces.append(binary_image)
    return processed_pieces

def save_images(image_pieces: List[np.ndarray]) -> None:
    output_dir = "pieces_output"
    os.makedirs(output_dir, exist_ok=True)

    for i, piece in enumerate(image_pieces):
        output_path = os.path.join(output_dir, f"piece_{i}.png")
        cv2.imwrite(output_path, piece)

def list_to_board(lst: list) -> list[list: str]:
    return [lst[i*4 : i*4 + 4] for i in range(4)]


def on_press(key: Union[Key, KeyCode]) -> None: # callback function
    if key == keyboard.Key.enter:
        print('Enter key pressed. Current mouse position is:', mouse_controller.position)
        mouse_positions.append(mouse_controller.position)

if __name__ == '__main__':

    letter_points = {
        'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1, 'J': 8,
        'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1, 'S': 1, 'T': 1,
        'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
    }

    bonus_tiles = {
        'TL': [],
        'DL': [],
        'TW': [],
        'DW': []
    }

    mouse_positions = []
    mouse_controller = mouse.Controller()
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    while len(mouse_positions) < 2: pass

    top_left = mouse_positions[0]
    bottom_right = mouse_positions[1]
    print('First mouse position:', mouse_positions[0])
    print('Second mouse position:', mouse_positions[1])

    keyboard_listener.stop()

    region = capture_screen_region_for_comparison(top_left, bottom_right)
    #cv2.imshow('Captured screen region', region)
    #cv2.waitKey(0)

    image_pieces = split_image_into_4x4_grid(region)
    image_pieces = binary_image_pieces(image_pieces)
    save_images(image_pieces)


    print('Loading model...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device:', torch.cuda.get_device_name(0))
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    model.to(device)
    print('Model loaded.')


    letters = []
    for index, piece in enumerate(image_pieces):
        scores = compareAllImages(piece, "binary_letters")
        print(scores)
        most_similar_letter = get_most_similar_letter(scores)
        letters.append(most_similar_letter)
        print('Most similar letter:', most_similar_letter)

    board = list_to_board(letters)

    print('Board:', board)


    boggle = Boggle(board)
    solved = boggle.solve()
    print(solved)















