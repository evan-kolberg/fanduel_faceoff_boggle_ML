# only converts to grey scale and resizes. useful if the images have been annotated and have corresponding label files.


from PIL import Image
import os

input_folder_path = 'raw-boards/'
output_folder_path = 'pre-processed-boards/'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for filename in os.listdir(input_folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')): 
        input_image_path = os.path.join(input_folder_path, filename)
    
        output_image_path = os.path.join(output_folder_path, os.path.splitext(filename)[0] + '_grayscale.png')
        
        with Image.open(input_image_path) as img:
            grayscale_img = img.convert('L') 
            grayscale_img = grayscale_img.resize((1024, 1024), Image.Resampling.LANCZOS)
            grayscale_img.save(output_image_path) 
            print(f'Converted {input_image_path} to {output_image_path}')













