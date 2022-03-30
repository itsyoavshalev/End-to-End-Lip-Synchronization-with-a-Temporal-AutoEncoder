import numpy as np

raw_db_list_path = '../data/lrs2/raw/test.txt'
with open(raw_db_list_path) as f:
    files_list = f.readlines()

files_list = [x.strip() for x in files_list]

for item in files_list:
    folder_name, file_name = item.split('/')
    shifts_file_path = '../data/lrs2/gen/test/{}/{}_shift'.format(folder_name, file_name)
    shift = np.random.randint(-25, 26)
    np.save(shifts_file_path, shift)

print('done')
