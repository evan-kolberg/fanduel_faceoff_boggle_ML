from PIL import Image
import os
import re

def extract_numbers_from_filename(filename):
    return ''.join(re.findall(r'\d+', filename))

input_folder_path = 'raw-boards/'
output_folder_path = 'pre-processed-boards/'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for filename in os.listdir(input_folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')): 
        input_image_path = os.path.join(input_folder_path, filename)
        
        numeric_file_name = extract_numbers_from_filename(filename)
        output_image_path = os.path.join(output_folder_path, numeric_file_name + '_grayscale.png')
        
        with Image.open(input_image_path) as img:
            grayscale_img = img.convert('L')
            if 'icc_profile' in img.info:
                del img.info['icc_profile']
            grayscale_img = grayscale_img.resize((1024, 1024), Image.Resampling.LANCZOS)
            grayscale_img.save(output_image_path) 
            print(f'Converted {input_image_path} to {output_image_path}')
