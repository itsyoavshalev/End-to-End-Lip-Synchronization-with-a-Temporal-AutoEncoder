import os
from common import *
import sys
import cv2


def main(argv):
    raw_files_path = '/media/yoavs/storage_d/DeepSync/data/TIMIT/raw/volunteers/'

    os.chdir('..')
    db_config = ConfigParser("dbs_config.yaml")

    raw_validation_path = db_config.TIMIT.raw_validation_path
    raw_validation_list_path = db_config.TIMIT.raw_validation_list_path
    raw_validation_list_org_names_path = db_config.TIMIT.raw_validation_list_org_names_path

    validationn_list_file = open(raw_validation_list_path, "a")
    validationn_list_org_names_file = open(raw_validation_list_org_names_path, "a")

    for file in os.listdir(raw_files_path):
        current_path = '{}{}'.format(raw_files_path, file)
        counter = 1
        videos = np.sort(np.array(os.listdir(current_path)))

        for video in videos:
            current_file_path = '{}/{}'.format(current_path, video)
            base_dest_path = '{}{}'.format(raw_validation_path, file)

            if not os.path.isdir(base_dest_path):
                os.makedirs(base_dest_path)

            file_name = '{:05d}'.format(counter)

            destination_file_path = '{}/{}.mp4'.format(base_dest_path, file_name)

            vidcap = cv2.VideoCapture(current_file_path)
            v_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            normalization_width = 720
            normalization_height = 720
            crop_top_left_x = v_width // 2 - normalization_width // 2
            crop_top_left_y = v_height // 2 - normalization_height // 3
            pad_or_crop = 'crop={0}:{1}:{2}:{3}'.format(normalization_width, normalization_height, crop_top_left_x, crop_top_left_y)

            command = 'ffmpeg -i {0} -r {1} -ar {2} -vf \"{6}, scale={4}:{5}\" {3}'.format(
                current_file_path, db_config.general.target_video_fps, db_config.general.target_audio_rate, destination_file_path,
                160, 160, pad_or_crop)

            error = os.system(command)

            if error:
                msg = 'error while converting video'
                print(msg)
                raise Exception(msg)

            file_record = '{}/{}\n'.format(file, file_name)
            validationn_list_file.write(file_record)
            validationn_list_org_names_file.write('{}\n'.format(video))

            counter += 1

    validationn_list_file.close()
    validationn_list_org_names_file.close()

    with open(raw_validation_list_org_names_path) as f:
        org_names = f.readlines()

    with open(raw_validation_list_path) as f:
        paths = f.readlines()

    org_names = np.array([x.strip() for x in org_names])
    paths = np.array([x.strip() for x in paths])

    raw_validation_list_pairs_path = db_config.TIMIT.raw_validation_list_pairs_path
    pairs_file = open(raw_validation_list_pairs_path, "a")

    for index_0, (file_name_0, path_0) in enumerate(zip(org_names, paths)):
        for index_1, (file_name_1, path_1) in enumerate(zip(org_names, paths)):
            if index_1 <= index_0:
                continue

            if file_name_0.lower() == file_name_1.lower():
                pairs_file.write('{} {} {} {} {} {}\n'.format(index_0, index_1, path_0, path_1, file_name_0, file_name_1))

    pairs_file.close()


if __name__ == '__main__':
    main(sys.argv)