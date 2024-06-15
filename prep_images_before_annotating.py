# renames images, converts to greyscale, and then resizes them


from PIL import Image
import os
import random
import string

def generate_random_name(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

input_folder_path = 'raw-boards/'
output_folder_path = 'pre-processed-boards/'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for filename in os.listdir(input_folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')): 
        input_image_path = os.path.join(input_folder_path, filename)
        
        random_file_name = generate_random_name()
        output_image_path = os.path.join(output_folder_path, random_file_name + '_grayscale.png')
        
        with Image.open(input_image_path) as img:
            grayscale_img = img.convert('L') 
            grayscale_img = grayscale_img.resize((1024, 1024), Image.Resampling.LANCZOS)
            grayscale_img.save(output_image_path) 
            print(f'Converted {input_image_path} to {output_image_path}')





