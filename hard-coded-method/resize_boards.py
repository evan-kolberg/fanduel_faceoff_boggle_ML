from PIL import Image
import os

def resize_images(folder_path, target_size=(603, 603)):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(file_path)

if __name__ == "__main__":
    boards_folder = "assets/boards"
    print(f"Resizing images in folder: {boards_folder}")
    resize_images(boards_folder)
    print("Images resized successfully!")
