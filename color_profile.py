import os
from PIL import Image

def remove_icc_profile(directory):
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Check if the file is a PNG image
        if os.path.isfile(file_path) and filename.lower().endswith('.png'):
            # Open the image file
            with Image.open(file_path) as img:
                # Check if the image has an 'icc_profile' attribute
                if 'icc_profile' in img.info:
                    # Remove the ICC profile
                    del img.info['icc_profile']
                    # Save the image without the ICC profile
                    img.save(file_path, "PNG")

# Replace 'path_to_directory' with the path to your directory containing PNG images
remove_icc_profile('dataset/train/images')
remove_icc_profile('dataset/val/images')




