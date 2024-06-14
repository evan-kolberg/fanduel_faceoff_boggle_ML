from sklearn.model_selection import train_test_split
import os
import shutil

def move_files_to_folder(files_list, destination_path):
    for f in files_list:
        try:
            shutil.move(f, destination_path)
        except:
            print(f)
            assert False


# Read images and annotations
images = [os.path.join('pre-processed-boards/', x) for x in os.listdir('pre-processed-boards/') if x[-3:] == "png"]
annotations = [os.path.join('annotations/', x) for x in os.listdir('annotations/') if x[-3:] == "txt"]

# Use sklearn function to shuffle and split samples into train and val sets.
train_images, val_images, train_annotations, val_annotations = train_test_split(images,
                                                                                annotations,
                                                                                test_size=0.2,
                                                                                random_state=42)

# Move the image splits into their folders
move_files_to_folder(train_images, 'dataset/train/images/')
move_files_to_folder(val_images, 'dataset/val/images')
# Move the annotation splits into their corresponding folders
move_files_to_folder(train_annotations, 'dataset/train/labels')
move_files_to_folder(val_annotations, 'dataset/val/labels')



