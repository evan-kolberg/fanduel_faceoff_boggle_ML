import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from paddleocr import PaddleOCR

# Initialize PaddleOCR for English and Italian
ocr_en = PaddleOCR(use_angle_cls=True, lang="en")
ocr_it = PaddleOCR(use_angle_cls=True, lang="it")

img_path = 'test3.png'

# Open the image and get its dimensions
image = Image.open(img_path).convert('RGB')
img_width, img_height = image.size

# Define cell dimensions
cell_width = img_width // 4
cell_height = img_height // 4

# Create a new image to stitch the results
stitched_image = Image.new('RGB', (img_width, img_height))

# Initialize a list to store all detections
all_detections = []

# Function to isolate black text in an image
def isolate_black_text(image):
    # Apply Gaussian blur to the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=1))
    # Convert the image to grayscale
    gray_image = blurred_image.convert('L')
    # Convert grayscale to binary (black and white)
    binary_image = gray_image.point(lambda p: p < 128 and 255)
    # Convert binary image back to RGB with black text on white background
    return binary_image.convert('RGB')

# Function to isolate white text in an image
def isolate_white_text(image):
    # Apply Gaussian blur to the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=1))
    # Convert the image to grayscale
    gray_image = blurred_image.convert('L')
    # Convert grayscale to binary (black and white) inverted
    binary_image = gray_image.point(lambda p: p > 128 and 255)
    # Convert binary image back to RGB with white text on black background
    return binary_image.convert('RGB')

# Process each cell in the grid
for i in range(4):
    for j in range(4):
        left = j * cell_width
        top = i * cell_height
        right = (j + 1) * cell_width
        bottom = (i + 1) * cell_height
        
        cell_img = image.crop((left, top, right, bottom))
        
        # First Pass: Detect only black letters
        black_img = isolate_black_text(cell_img)
        result = ocr_en.ocr(np.array(black_img), cls=True)
        detected_letters = []
        if result:
            for line in result:
                if line:
                    for word in line:
                        if word:
                            text = word[1][0]
                            detected_letters.append(text.upper())
        
        # If no letter detected on the first pass, try Italian OCR
        if not detected_letters:
            result = ocr_it.ocr(np.array(black_img), cls=True)
            if result:
                for line in result:
                    if line:
                        for word in line:
                            if word:
                                text = word[1][0]
                                detected_letters.append(text.upper())
        
        # Only proceed to the second pass if all cells have been processed with the first pass
        if all_detections:
            # Second Pass: Detect only white special conditions (DL, TL, TW, DW)
            white_img = isolate_white_text(cell_img)
            result = ocr_en.ocr(np.array(white_img), cls=True)
            special_conditions = []
            if result:
                for line in result:
                    if line:
                        for word in line:
                            if word:
                                text = word[1][0]
                                special_conditions.append(text)
            
            # Combine both OCR detections
            detections = detected_letters + special_conditions
            all_detections.append(detections)
        else:
            all_detections.append(detected_letters)
        
        # Draw bounding boxes on the cell image
        draw = ImageDraw.Draw(cell_img)
        for line in result:
            if line:
                for word in line:
                    if word:
                        box = word[0]
                        x0, y0 = box[0]
                        x1, y1 = box[2]
                        box_tuple = (x0, y0, x1, y1)
                        draw.rectangle(box_tuple, outline='red')
        
        # Paste the processed cell image back into the stitched image
        stitched_image.paste(cell_img, (left, top))

# Save the stitched image
stitched_image.save('stitched_result.jpg')

print(all_detections)
