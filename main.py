import subprocess
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")
img_path = 'test3.png'

# Open the image and get its dimensions
image = Image.open(img_path).convert('RGB')
img_width, img_height = image.size

# Calculate dimensions of each cell in the 4x4 grid
cell_width = img_width // 4
cell_height = img_height // 4

# Create a blank image to stitch the results
stitched_image = Image.new('RGB', (img_width, img_height))

# Loop through the 4x4 grid
for i in range(4):
    for j in range(4):
        # Calculate the coordinates of the current cell
        left = j * cell_width
        top = i * cell_height
        right = (j + 1) * cell_width
        bottom = (i + 1) * cell_height
        
        # Crop the current cell from the image
        cell_img = image.crop((left, top, right, bottom))
        
        # Perform OCR on the current cell
        result = ocr.ocr(np.array(cell_img), cls=True)
        
        # Check if OCR result is not None
        if result is not None:
            # Create a drawing object
            draw = ImageDraw.Draw(cell_img)
            
            # Draw bounding boxes on the cell image
            for line in result:
                # Check if line is not None
                if line is not None:
                    for word in line:
                        # Check if word is not None
                        if word is not None:
                            # Extract box coordinates
                            box = word[0]
                            # Convert box coordinates to tuple format (x0, y0, x1, y1)
                            x0, y0 = box[0]
                            x1, y1 = box[2]
                            box_tuple = (x0, y0, x1, y1)
                            draw.rectangle(box_tuple, outline='red')
            
            # Paste the cell image with bounding boxes onto the stitched image
            stitched_image.paste(cell_img, (left, top))

# Save the stitched image
stitched_image.save('stitched_result.jpg')
