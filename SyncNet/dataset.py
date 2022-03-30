from __future__ import print_function
import torch
import numpy as np
import torch.utils.data as data
import os
from imgaug import augmenters as iaa
import random
from common import DBType, extract_mouth_from_frame
from scipy.io import wavfile
import python_speech_features
import cv2
from PIL import Image


class VideoMetadata:
    pass


class FrameMetadata:
    pass


def init_dataset(config, db_config, db_type):
    if not db_config.general.db_name == 'lrs2_dataset':
        raise Exception('this type of dataset is not supported yet')

    if db_type == DBType.Example:
        dataset = LRS2Dataset(db_config, config, db_type, False)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=0, drop_last=False)
    elif db_type == DBType.Train:
        dataset = LRS2Dataset(db_config, config, db_type, False)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, drop_last=False)
    else:
        dataset = LRS2Dataset(db_config, config, db_type, True)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.test.batch_size, shuffle=False, num_workers=config.test.num_workers, drop_last=False)

    return data_loader, dataset


class LRS2Dataset(data.Dataset):
    def __init__(self, db_config, config, db_type, is_test):
        self.is_test = is_test
        lrs2_config = eval('db_config.{}'.format(db_config.general.db_name))
        self.db_type = db_type

        self.rgb2gray_augmentor = iaa.Sequential([
            iaa.Grayscale(alpha=1.0)
        ]).to_deterministic()

        self.image_channels_augmentor = iaa.Sequential([
            iaa.Multiply((0.5, 1.5), per_channel=1)
        ])

        # distortion_prob = lambda aug: iaa.Sometimes(0.7, aug)
        
        self.distortion_augmentor = iaa.Sequential([
            #distortion_prob(
                iaa.PiecewiseAffine(scale=(0.025, 0.05))
            #)
            ])

        # affine_prob = lambda aug: iaa.Sometimes(0.8, aug)
        # always = lambda aug: iaa.Sometimes(1, aug)

        self.affine_augmentor = iaa.Sequential(
            [
                #always(
                    iaa.Fliplr(0.5)
                #)
                ,
                # sometimes(iaa.Dropout(p=(0, 0.3))),
                #always(
                    iaa.Affine(
                    scale=({"x": (0.9, 1.1), "y": (0.9, 1.1)}),  # scale images to % of their size
                    rotate=(-5, 5),  # rotate by degrees
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate percent (per axis)
                    # shear=(-16, 16), # shear by degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=128, # if mode is constant, use a cval between 0 and 255
                    # mode="edge"# use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    # ))
                    # ,
                    # self.sometimes(iaa.CropAndPad(
                    # percent=([0, -0.05], [0, -0.05], [0, -0.05], [0, -0.05]),
                    # pad_mode=ia.ALL,
                    # pad_cval=(0, 255)
                #)
                    )]
            # ,random_order=True
        )

        # image_dropout_prob = lambda aug: iaa.Sometimes(0.5, aug)
        self.image_dropout = iaa.Sequential([
            #image_dropout_prob(
                iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
            #)
        ])

        if db_type == DBType.Test:
            self.base_path = lrs2_config.test_path
            self.batch_size = config.test.batch_size
            raw_db_list_path = lrs2_config.raw_test_list_path
        elif db_type == DBType.Validation:
            self.base_path = lrs2_config.validation_path
            self.batch_size = config.test.batch_size
            raw_db_list_path = lrs2_config.raw_validation_list_path
        elif db_type == DBType.Train:
            self.base_path = lrs2_config.train_path
            self.batch_size = config.train.batch_size
            raw_db_list_path = lrs2_config.raw_train_list_path
        elif db_type == DBType.Example:
            self.base_path = lrs2_config.example_path
            self.batch_size = config.train.batch_size
            raw_db_list_path = lrs2_config.raw_example_list_path
        else:
            raise Exception('Invalid db type')

        if not self.is_test:
            self.tmp_folder = '{}/tmp/'.format(config.general.output_path)
            os.makedirs(self.tmp_folder, exist_ok=True)

        with open(raw_db_list_path) as f:
            files_list = f.readlines()

        files_list = np.array([x.strip() for x in files_list])

        self.video_fps = np.load('{}{}.npy'.format(self.base_path, lrs2_config.video_fps_file_name))
        self.audio_rate = np.load('{}{}.npy'.format(self.base_path, lrs2_config.audio_rate_file_name))
        self.ef_frames_window_size = config.general.ef_frames_window_size
        self.audio_window_size = config.general.audio_window_size
        self.mouth_height = db_config.general.mouth_height
        self.mouth_width = db_config.general.mouth_width


        # each video frame is 40 ms (for 25 fps)
        video_frame_duration_ms = 1000 / self.video_fps

        # each audio time step is 10 ms (for 16000 khz, sampling rate of 100Hz for MFCC)
        audio_time_step_ms = 10

        self.audio_video_ratio = int(video_frame_duration_ms // audio_time_step_ms)

        assert self.audio_video_ratio == (video_frame_duration_ms / audio_time_step_ms)

        if not self.is_test:
            self.frames_shift_window_train = config.general.frames_shift_window_train
            self.audio_shift_window = self.frames_shift_window_train * self.audio_video_ratio

        tmp_videos_md = []
        tmp_frames_metadata = []

        for file in files_list:
            folder_name, file_name = file.split('/')

            current_video_frames_file_path = '{0}{1}/{2}_{3}.mp4'.format(self.base_path, folder_name, file_name, lrs2_config.video_frames_file_name)
            current_video_frames_path = '{0}{1}/{2}_frames'.format(self.base_path, folder_name, file_name)
            current_video_anchors_path = '{0}{1}/{2}_{3}.npy'.format(self.base_path, folder_name, file_name, lrs2_config.anchors_file_name)
            current_audio_features_path = '{0}{1}/{2}_af.npy'.format(self.base_path, folder_name, file_name)

            if not os.path.isfile(current_video_frames_file_path):
                continue

            num_of_frames_file_path = '{}{}/{}_{}.npy'.format(self.base_path, folder_name, file_name,
                                                              lrs2_config.num_of_frames_file_name)

            num_of_audio_frames_file_path = '{}{}/{}_{}.npy'.format(self.base_path, folder_name, file_name,
                                                              lrs2_config.num_of_audio_frames_file_name)

            number_of_frames_in_video = np.load(num_of_frames_file_path)
            number_of_audio_frames = np.load(num_of_audio_frames_file_path)

            current_video_md = VideoMetadata()
            current_video_md.video_frames_path = current_video_frames_path
            current_video_md.anchors_path = current_video_anchors_path

            current_video_md.audio_features_path = current_audio_features_path
            current_video_md.number_of_frames_in_video = number_of_frames_in_video
            current_video_md.file_name = file_name
            current_video_md.folder_name = folder_name
            current_video_md.number_of_audio_frames = number_of_audio_frames

            if self.is_test:
                current_video_shift_path = '{}{}/{}_shift.npy'.format(self.base_path, folder_name, file_name)
                current_video_md.shift = np.load(current_video_shift_path)
            else:
                current_video_md.shift = 0

            tmp_videos_md.append(current_video_md)

            for local_id in range(0, number_of_frames_in_video - self.ef_frames_window_size + 1):
                max_audio_index = local_id * self.audio_video_ratio + self.audio_window_size

                if max_audio_index > number_of_audio_frames:
                    break

                frame_md = FrameMetadata()
                frame_md.local_id = local_id
                frame_md.video_md = current_video_md

                tmp_frames_metadata.append(frame_md)

        self.num_of_global_frames = len(tmp_frames_metadata)

        self.videos_md = np.ndarray((len(tmp_videos_md),), dtype=np.object)
        for i in range(0, len(tmp_videos_md)):
            self.videos_md[i] = tmp_videos_md[i]

        self.frames_metadata = np.ndarray((len(tmp_frames_metadata),), dtype=np.object)
        for i in range(0, len(tmp_frames_metadata)):
            self.frames_metadata[i] = tmp_frames_metadata[i]

    def __len__(self):
        return self.num_of_global_frames // self.batch_size * self.batch_size

    def __getitem__(self, global_frame_index):
        return self.load_one_train_set(global_frame_index)

    def get_number_of_videos(self):
        return len(self.videos_md)

    def load_video_inputs(self, video_id):
        video_md = self.videos_md[video_id]

        number_of_frames_in_video = video_md.number_of_frames_in_video
        number_of_audio_frames = video_md.number_of_audio_frames

        anchors = np.load(video_md.anchors_path)
        assert len(anchors) == number_of_frames_in_video

        shift = video_md.shift

        # frames are saved as RGB
        # video_cap = cv2.VideoCapture(video_md.video_frames_path)
        # total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #
        # assert total_frames == number_of_frames_in_video

        audio_features_path = '{}{}/{}_af.npy'.format(self.base_path, video_md.folder_name, video_md.file_name)
        input_audio_features = np.load(audio_features_path)
        assert input_audio_features.shape == (13, number_of_audio_frames)

        input_frames = []
        for i_frame in range(0, number_of_frames_in_video):
            # video_cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            # success, frame = video_cap.read()
            #
            # if not success:
            #     raise Exception('Error while reading frame')
            frame_path = '{}/{:0>5d}.jpg'.format(video_md.video_frames_path, (i_frame + 1))
            #frame = cv2.imread(frame_path)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with Image.open(frame_path) as pi:
                frame = np.array(pi)

            mouth = extract_mouth_from_frame(frame, anchors[i_frame], self.mouth_height, self.mouth_width)
            mouth = np.expand_dims(mouth, axis=0)
            mouth = self.rgb2gray_augmentor.augment_images(mouth)
            input_frames.append(mouth[0, :, :, 0])

        input_frames = np.array(input_frames)
        assert len(input_frames) == number_of_frames_in_video

        input_audio_features = torch.tensor(input_audio_features, dtype=torch.float)
        input_frames = torch.tensor(input_frames, dtype=torch.float)

        input_dict = {'video_frames': input_frames,
                      'audio_features': input_audio_features,
                      'video_frames_shift': shift}

        #video_cap.release()

        return input_dict

    def load_one_train_set(self, global_frame_index):
        if self.is_test:
            raise Exception('This method shouldn\'t be called at test time')

        frame_md = self.frames_metadata[global_frame_index]

        video_md = frame_md.video_md

        local_frame_index = frame_md.local_id
        number_of_frames_in_video = video_md.number_of_frames_in_video
        number_of_audio_frames = video_md.number_of_audio_frames

        anchors = np.load(video_md.anchors_path)
        assert len(anchors) == number_of_frames_in_video

        # frames are saved as RGB
        #video_cap = cv2.VideoCapture(video_md.video_frames_path)
        #total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        #assert total_frames == number_of_frames_in_video

        augment_volume = False
        if augment_volume:
            audio_file_path = '{}{}/con_{}.wav'.format(self.base_path, video_md.folder_name, video_md.file_name)

            augmented_audio_path = '{}{}.wav'.format(self.tmp_folder, global_frame_index)

            if os.path.isfile(augmented_audio_path):
                os.remove(augmented_audio_path)

            p = random.uniform(0.9, 1.1)
            error = os.system('ffmpeg -loglevel error -hwaccel cuvid -i {} -filter:a "volume={}" {}'.format(audio_file_path,
                                                                             p,
                                                                             augmented_audio_path))
            if error:
                raise Exception('error while augmenting audio')

            sample_rate, audio = wavfile.read(augmented_audio_path)
            mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
            audio_features = np.stack([np.array(i) for i in mfcc]).astype(float)
            os.remove(augmented_audio_path)
        else:
            audio_features_path = '{}{}/{}_af.npy'.format(self.base_path, video_md.folder_name, video_md.file_name)
            audio_features = np.load(audio_features_path)

        assert audio_features.shape == (13, number_of_audio_frames)

        affine_augmentor = self.affine_augmentor.to_deterministic()
        image_channels_augmentor = self.image_channels_augmentor.to_deterministic()

        input_frames = []

        # augmentation is executed one by one, in order to use deterministic augmentation (otherwise you will get a random aug for each image (imgaug library))
        for i_frame in range(local_frame_index, (local_frame_index + self.ef_frames_window_size)):
            #video_cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            #success, frame = video_cap.read()
            frame_path = '{}/{:0>5d}.jpg'.format(video_md.video_frames_path, (i_frame+1))
            #frame = cv2.imread(frame_path)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with Image.open(frame_path) as pi:
                frame = np.array(pi)

            #if not success:
            #    raise Exception('Error while reading frame')

            mouth = extract_mouth_from_frame(frame, anchors[i_frame], self.mouth_height, self.mouth_width)
            mouth = np.expand_dims(mouth, axis=0)
            mouth = affine_augmentor.augment_images(mouth)
            mouth = image_channels_augmentor.augment_images(mouth)
            mouth = self.image_dropout.augment_images(mouth)
            mouth = self.rgb2gray_augmentor.augment_images(mouth)
            input_frames.append(mouth[0, :, :, 0])

        input_frames = np.array(input_frames)
        assert len(input_frames) == self.ef_frames_window_size

        # for i in range(0, len(input_frames)):
        #     cv2.imwrite('c://temp//{}.png'.format(i), input_frames[i])

        org_start_audio_index = local_frame_index * self.audio_video_ratio

        if np.random.rand() > 0.5:
            start_audio_index = org_start_audio_index
            end_audio_index = start_audio_index + self.audio_window_size
        else:
            # the first audio feature can be shifted 2 seconds back
            min_start_audio_index = max(0, org_start_audio_index - self.audio_shift_window)

            # or 2 seconds forward, exclusive
            max_start_audio_index = min(org_start_audio_index + self.audio_shift_window, audio_features.shape[1]-self.audio_window_size+1)

            start_audio_index = np.random.randint(min_start_audio_index, max_start_audio_index)
            end_audio_index = start_audio_index + self.audio_window_size

        input_audio_features = audio_features[:, start_audio_index:end_audio_index]

        if start_audio_index == org_start_audio_index:
            target_prediction = 1
        else:
            target_prediction = 0

        target_prediction = torch.tensor(target_prediction, dtype=torch.float)
        input_audio_features = torch.tensor(input_audio_features, dtype=torch.float).unsqueeze(0)
        input_frames = torch.tensor(input_frames, dtype=torch.float).unsqueeze(0)

        input_dict = {'visual_input': input_frames,
                      'target_prediction': target_prediction,
                      'audio_features': input_audio_features}

        # video_cap.release()

        return input_dict
