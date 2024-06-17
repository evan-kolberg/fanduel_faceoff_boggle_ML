import os
import re

def extract_numbers_from_filename(filename):
    return ''.join(re.findall(r'\d+', filename))

input_folder_path = 'obj_train_data'
output_folder_path = 'obj_train_data'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for filename in os.listdir(input_folder_path):
    if filename.endswith('.txt'):
        input_txt_path = os.path.join(input_folder_path, filename)
        
        numeric_file_name = extract_numbers_from_filename(filename)
        output_txt_path = os.path.join(output_folder_path, numeric_file_name + '_grayscale.txt')
        
        os.rename(input_txt_path, output_txt_path)
        print(f'Renamed {input_txt_path} to {output_txt_path}')



