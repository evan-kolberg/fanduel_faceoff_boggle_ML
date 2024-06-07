from PIL import Image
import os

def resize_images(folder_path, target_size=(150, 150)):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            img_resized = img.resize(target_size, Image.LANCZOS)  # Using LANCZOS filter
            img_resized.save(file_path)

if __name__ == "__main__":
    letters_folder = "assets/letters"
    print(f"Resizing images in folder: {letters_folder}")
    resize_images(letters_folder)
    print("Images resized successfully!")
