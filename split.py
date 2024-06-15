# does an 80-20 split of the dataset into training and validation sets


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


images = [os.path.join('pre-processed-boards/', x) for x in os.listdir('pre-processed-boards/') if x[-3:] == "png"]
annotations = [os.path.join('obj_train_data/', x) for x in os.listdir('obj_train_data/') if x[-3:] == "txt"]

train_images, val_images, train_annotations, val_annotations = train_test_split(images,
                                                                                annotations,
                                                                                test_size=0.2,
                                                                                random_state=42)

move_files_to_folder(train_images, 'dataset/train/images/')
move_files_to_folder(val_images, 'dataset/val/images/')

move_files_to_folder(train_annotations, 'dataset/train/labels/')
move_files_to_folder(val_annotations, 'dataset/val/labels/')



