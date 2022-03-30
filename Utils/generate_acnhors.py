import numpy as np
import os

#raw_db_list_path = '../data/lrs2/raw/val.txt'
raw_db_list_path = '../data/TIMIT/raw/val.txt'

with open(raw_db_list_path) as f:
    files_list = f.readlines()

files_list = [x.strip() for x in files_list]

for item in files_list:
    folder_name, file_name = item.split('/')
    #shapes_file_path = '../data/lrs2/gen/validation/{}/{}_shapes_2d.npy'.format(folder_name, file_name)
    #anchors_file_path = '../data/lrs2/gen/validation/{}/{}_anchors.npy'.format(folder_name, file_name)
    shapes_file_path = '../data/TIMIT/gen/validation/{}/{}_shapes_2d.npy'.format(folder_name, file_name)
    anchors_file_path = '../data/TIMIT/gen/validation/{}/{}_anchors.npy'.format(folder_name, file_name)
    if os.path.isfile(shapes_file_path):
        shapes = np.load(shapes_file_path)
        anchors = shapes[:, 29]
        np.save(anchors_file_path, anchors)

print('done')
