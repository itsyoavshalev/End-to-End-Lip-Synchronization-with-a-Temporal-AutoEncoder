import shutil
import os
import numpy as np
import shutil
from numpy import genfromtxt
from common import *
import multiprocessing
from multiprocessing import Pool


#raw_db_list_path = '../data/lrs2/raw/pretrain.txt'
#raw_db_list_path = '../data/lrs2/raw/test.txt'
#raw_db_list_path = '../data/lrs2/raw/val.txt'
raw_db_list_path = '../data/TIMIT/raw/val.txt'


def run_wrapper(args):
    return run(*args)

def run(item, i):
    print(i)
    folder_name, file_name = item.split('/')
    #file_path = '../data/lrs2/gen/pretrain/{}/{}_video_frames.mp4'.format(folder_name, file_name)
    #file_path = '../data/lrs2/gen/test/{}/{}_video_frames.mp4'.format(folder_name, file_name)
    #file_path = '../data/lrs2/gen/validation/{}/{}_video_frames.mp4'.format(folder_name, file_name)
    file_path = '../data/TIMIT/gen/validation/{}/{}_video_frames.mp4'.format(folder_name, file_name)


    if not os.path.isfile(file_path):
        return

    #dest_path = '../data/lrs2/gen/pretrain/{}/{}_frames'.format(folder_name, file_name)
    #dest_path = '../data/lrs2/gen/test/{}/{}_frames'.format(folder_name, file_name)
    #dest_path = '../data/lrs2/gen/validation/{}/{}_frames'.format(folder_name, file_name)
    dest_path = '../data/TIMIT/gen/validation/{}/{}_frames'.format(folder_name, file_name)

    # if os.path.isdir(dest_path):
    #     shutil.rmtree(dest_path)
    #
    # return

    os.mkdir(dest_path)
    frames_path = '{}/%05d.jpg'.format(dest_path)

    error = os.system(
        'ffmpeg -loglevel error -hwaccel cuvid -i {} -qscale:v 5 {}'.format(
            file_path,
            frames_path
        ))

    if error:
        msg = 'error while generating frames'
        print(msg)
        raise Exception(msg)


with open(raw_db_list_path) as f:
    files_list = f.readlines()

files_list = [x.strip() for x in files_list]

num_cores = multiprocessing.cpu_count()

with Pool(num_cores) as pool:
    params_p = [(item, index)
                for index, item in
                enumerate(files_list)]

    pool.map(run_wrapper, params_p)

print('done')
