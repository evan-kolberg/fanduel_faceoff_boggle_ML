from PIL import Image
import imagehash
import os

def compare_images(input_image_path, image_directory):
    input_hash = imagehash.average_hash(Image.open(input_image_path))

    similar_images = []
    for filename in os.listdir(image_directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_directory, filename)
            other_hash = imagehash.average_hash(Image.open(image_path))
            similarity_score = input_hash - other_hash
            similar_images.append((filename, similarity_score))

    similar_images.sort(key=lambda x: x[1])

    if similar_images:
        print(f"Most similar image: {similar_images[0][0]} (Score: {similar_images[0][1]})")
    else:
        print("No similar images found in the directory.")

if __name__ == "__main__":
    input_image_path = "hard-coded-method/output/blurred_S.png"
    image_directory = "hard-coded-method/output"
    compare_images(input_image_path, image_directory)



