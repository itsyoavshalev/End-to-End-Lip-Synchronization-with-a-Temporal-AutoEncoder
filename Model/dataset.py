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
from enum import Enum


class VideoMetadata:
    pass


class FrameMetadata:
    pass


class FrameAction(Enum):
    Keep = 0,
    Drop = 1,
    Duplicate = 2


def init_dataset(config, db_config, db_type, is_single_shift):
    print('Using {}'.format(db_config.general.db_name))

    if db_type == DBType.Example:
        if not db_config.general.db_name == 'lrs2_dataset':
            raise Exception('this type of dataset is not supported yet')

        dataset = Dataset(db_config, config, db_type, False, is_single_shift)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=0, drop_last=False)
    elif db_type == DBType.Train:
        if not db_config.general.db_name == 'lrs2_dataset':
            raise Exception('this type of dataset is not supported yet')

        dataset = Dataset(db_config, config, db_type, False, is_single_shift)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, drop_last=False)
    else:
        if is_single_shift and not db_config.general.db_name == 'lrs2_dataset':
            raise Exception('this type of dataset is not supported yet')

        dataset = Dataset(db_config, config, db_type, True, is_single_shift)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.test.batch_size, shuffle=False, num_workers=config.test.num_workers, drop_last=False)

    return data_loader, dataset


def assert_prediction(predicted_frame, original_frame, frame_action):
    if not frame_action == FrameAction.Drop:
        assert (predicted_frame.data == original_frame).min()


class Dataset(data.Dataset):
    def __init__(self, db_config, config, db_type, is_test, is_single_shift, gen_mode=False):
        if is_single_shift and not db_config.general.db_name == 'lrs2_dataset':
            raise Exception('this type of dataset is not supported yet')

        self.is_test = is_test
        self.db_type = db_type
        self.config = config
        self.is_single_shift = is_single_shift
        self.gen_mode = gen_mode
        self.db_config = db_config

        specific_db_config = eval('db_config.{}'.format(db_config.general.db_name))
        self.db_folder_name = specific_db_config.db_folder_name

        if db_type == DBType.Test:
            if self.is_single_shift or gen_mode:
                self.base_path = specific_db_config.test_path
                raw_db_list_path = specific_db_config.raw_test_list_path
            else:
                self.base_path = './data/{}/gen/test_ma/'.format(self.db_folder_name)

            self.batch_size = config.test.batch_size
        elif db_type == DBType.Validation:
            if self.is_single_shift or gen_mode:
                self.base_path = specific_db_config.validation_path
                raw_db_list_path = specific_db_config.raw_validation_list_path
            else:
                self.base_path = './data/{}/gen/val_ma/'.format(self.db_folder_name)

            self.batch_size = config.test.batch_size
        elif db_type == DBType.Train:
            if not db_config.general.db_name == 'lrs2_dataset':
                raise Exception('this type of dataset is not supported yet')

            self.base_path = specific_db_config.train_path
            self.batch_size = config.train.batch_size
            raw_db_list_path = specific_db_config.raw_train_list_path
        elif db_type == DBType.Example:
            if not db_config.general.db_name == 'lrs2_dataset':
                raise Exception('this type of dataset is not supported yet')

            self.base_path = specific_db_config.example_path
            self.batch_size = config.train.batch_size
            raw_db_list_path = specific_db_config.raw_example_list_path
        else:
            raise Exception('Invalid db type')

        if self.db_folder_name == 'TIMIT':
            assert self.batch_size == 1

        self.padding_value = config.general.max_seq_len + 1
        self.max_seq_len = config.general.max_seq_len
        self.sync_len = self.config.general.sync_len

        if is_test and not is_single_shift and not gen_mode:
            self.db_len = len(os.listdir(self.base_path))

            if self.db_folder_name == 'TIMIT':
                pairs_file_path = specific_db_config.raw_validation_list_pairs_path
                with open(pairs_file_path) as pairs_file:
                    pairs_list = pairs_file.readlines()

                self.pairs_list = np.array([x.strip() for x in pairs_list])
                self.db_len = len(self.pairs_list)
        else:
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

            with open(raw_db_list_path) as f:
                files_list = f.readlines()

            files_list = np.array([x.strip() for x in files_list])

            self.video_fps = np.load('{}{}.npy'.format(self.base_path, specific_db_config.video_fps_file_name))
            self.audio_rate = np.load('{}{}.npy'.format(self.base_path, specific_db_config.audio_rate_file_name))
            self.ef_frames_window_size = config.general.ef_frames_window_size
            self.audio_window_size = config.general.audio_window_size
            self.mouth_height = db_config.general.mouth_height
            self.mouth_width = db_config.general.mouth_width

            # each video frame is 40 ms (for 25 fps)
            video_frame_duration_ms = 1000 / self.video_fps

            # each audio time step is 10 ms (for 16000 khz, sampling rate of 100Hz for MFCC)
            audio_time_step_ms = 10

            self.audio_video_ratio = int(video_frame_duration_ms // audio_time_step_ms)

            assert video_frame_duration_ms % audio_time_step_ms == 0
            assert self.audio_window_size % self.audio_video_ratio == 0

            self.frames_shift_window_train = config.general.frames_shift_window_train
            self.max_frame_duplications = config.general.max_frame_duplications

            tmp_videos_md = []

            for file in files_list:
                folder_name, file_name = file.split('/')

                current_video_frames_file_path = '{0}{1}/{2}_{3}.mp4'.format(self.base_path, folder_name, file_name, specific_db_config.video_frames_file_name)
                current_video_frames_path = '{0}{1}/{2}_frames'.format(self.base_path, folder_name, file_name)
                current_video_anchors_path = '{0}{1}/{2}_{3}.npy'.format(self.base_path, folder_name, file_name, specific_db_config.anchors_file_name)
                current_audio_features_path = '{0}{1}/{2}_af.npy'.format(self.base_path, folder_name, file_name)

                if not os.path.isfile(current_video_frames_file_path):
                    continue

                num_of_frames_file_path = '{}{}/{}_{}.npy'.format(self.base_path, folder_name, file_name,
                                                                  specific_db_config.num_of_frames_file_name)

                num_of_audio_frames_file_path = '{}{}/{}_{}.npy'.format(self.base_path, folder_name, file_name,
                                                                  specific_db_config.num_of_audio_frames_file_name)

                number_of_frames_in_video = np.load(num_of_frames_file_path)
                number_of_audio_frames = np.load(num_of_audio_frames_file_path)

                effective_num_of_video_frames = number_of_frames_in_video - (self.ef_frames_window_size - 1)
                effective_num_of_audio_frames = number_of_audio_frames // self.audio_video_ratio - (
                        self.audio_window_size // self.audio_video_ratio - 1)

                min_effective_num_of_frames = min(effective_num_of_video_frames, effective_num_of_audio_frames)

                if min_effective_num_of_frames < self.sync_len:
                    continue

                current_video_md = VideoMetadata()

                current_video_md.min_effective_num_of_frames = min_effective_num_of_frames
                current_video_md.video_frames_path = current_video_frames_path
                current_video_md.anchors_path = current_video_anchors_path

                current_video_md.audio_features_path = current_audio_features_path
                current_video_md.number_of_frames_in_video = number_of_frames_in_video
                current_video_md.file_name = file_name
                current_video_md.folder_name = folder_name
                current_video_md.number_of_audio_frames = number_of_audio_frames

                if self.is_test and is_single_shift:
                    current_video_shift_path = '{}{}/{}_shift.npy'.format(self.base_path, folder_name, file_name)
                    current_video_md.test_shift = np.load(current_video_shift_path)
                else:
                    current_video_md.test_shift = 0

                tmp_videos_md.append(current_video_md)

            self.videos_md = np.array(tmp_videos_md)
            self.db_len = len(self.videos_md) // self.batch_size * self.batch_size

    def __len__(self):
        return self.db_len

    def generate_test_db(self):
        if not self.gen_mode:
            raise Exception('Db is not in gen mode')

        if self.db_type == DBType.Validation:
            out_path = './data/{}/gen/val_ma/'.format(self.db_folder_name)
        elif self.db_type == DBType.Test:
            out_path = './data/{}/gen/test_ma/'.format(self.db_folder_name)
        else:
            raise Exception('Invalid db type')

        os.mkdir(out_path)

        for i in range(0, self.db_len):
            if self.db_folder_name == 'TIMIT':
                inputs = self.load_TIMIT_inputs(i)
            else:
                inputs = self.load_one_train_set(i)
            db_output_path = '{}{}'.format(out_path, i)
            np.savez_compressed(db_output_path, inputs)

    def __getitem__(self, index):
        if self.is_test:
            if self.is_single_shift:
                assert self.db_folder_name == 'lrs2'
                return self.load_one_test_set_single(index)
            else:
                if self.db_folder_name == 'TIMIT':
                    return self.load_one_test_set_TIMIT(index)
                else:
                    return self.load_one_test_set(index)
        else:
            assert self.db_folder_name == 'lrs2'

            if self.is_single_shift:
                return self.load_one_train_set_single(index)
            else:
                return self.load_one_train_set(index)

    def load_one_test_set_TIMIT(self, pair_index):
        pair = self.pairs_list[pair_index].split()
        index_0 = pair[0]
        index_1 = pair[1]

        dict_0 = self.load_test_dict(index_0)
        dict_1 = self.load_test_dict(index_1)

        visual_input_0_np = dict_0['visual_input']
        visual_input_1_np = dict_1['visual_input']

        if visual_input_0_np.shape[0] < self.max_seq_len:
            pad_size = self.max_seq_len - visual_input_0_np.shape[0]
            pad = np.zeros(pad_size * visual_input_0_np.shape[1] * visual_input_0_np.shape[2] * visual_input_0_np.shape[3]).reshape(
                (pad_size, *visual_input_0_np.shape[1:]))
            visual_input_0_np = np.concatenate((visual_input_0_np, pad))
        if visual_input_1_np.shape[0] < self.max_seq_len:
            pad_size = self.max_seq_len - visual_input_1_np.shape[0]
            pad = np.zeros(pad_size * visual_input_1_np.shape[1] * visual_input_1_np.shape[2] * visual_input_1_np.shape[3]).reshape(
                (pad_size, *visual_input_1_np.shape[1:]))
            visual_input_1_np = np.concatenate((visual_input_1_np, pad))

        audio_features_0 = torch.tensor(dict_0['audio_features'], dtype=torch.float)
        visual_input_0 = torch.tensor(visual_input_0_np, dtype=torch.float)

        audio_features_1 = torch.tensor(dict_1['audio_features'], dtype=torch.float)
        visual_input_1 = torch.tensor(visual_input_1_np, dtype=torch.float)

        raw_volunteers_path = self.db_config.TIMIT.raw_volunteers_path
        validation_path = self.db_config.TIMIT.validation_path

        input_dict = {'visual_input_0': visual_input_0,
                      'audio_features_0': audio_features_0,
                      'visual_input_1': visual_input_1,
                      'audio_features_1': audio_features_1,
                      'video_path_0': '{}{}/{}'.format(raw_volunteers_path, pair[2].split('/')[0], pair[4]),
                      'video_path_1': '{}{}/{}'.format(raw_volunteers_path, pair[3].split('/')[0], pair[5]),
                      'audio_path_0': '{}{}/con_{}.wav'.format(validation_path, pair[2].split('/')[0], pair[2].split('/')[1]),
                      'audio_path_1': '{}{}/con_{}.wav'.format(validation_path, pair[3].split('/')[0], pair[3].split('/')[1])
        }

        return input_dict

    def load_test_dict(self, index):
        path = '{}{}.npz'.format(self.base_path, index)
        data = np.load(path)
        dict = {key: data[key].item() for key in data}
        dict = dict['arr_0']
        return dict

    def load_one_test_set(self, video_index):
        dict = self.load_test_dict(video_index)

        audio_features = torch.tensor(dict['audio_features'], dtype=torch.float)
        visual_input = torch.tensor(dict['visual_input'], dtype=torch.float)

        target_prediction = torch.tensor(dict['target_prediction'], dtype=torch.long)
        video_seq_len = torch.tensor(dict['video_seq_len'], dtype=torch.float)
        audio_seq_len = torch.tensor(dict['audio_seq_len'], dtype=torch.float)

        input_dict = {'visual_input': visual_input,
                      'audio_features': audio_features,
                      'target_prediction': target_prediction,
                      'video_seq_len': video_seq_len,
                      'audio_seq_len': audio_seq_len}

        return input_dict

    def load_TIMIT_inputs(self, video_index):
        assert self.db_folder_name == 'TIMIT'

        video_md = self.videos_md[video_index]
        orig_number_of_frames_in_video = video_md.number_of_frames_in_video

        anchors = np.load(video_md.anchors_path)
        assert len(anchors) == orig_number_of_frames_in_video

        audio_features_path = '{}{}/{}_af.npy'.format(self.base_path, video_md.folder_name, video_md.file_name)
        audio_features = np.load(audio_features_path)

        orig_number_of_audio_frames = video_md.number_of_audio_frames
        assert audio_features.shape == (13, orig_number_of_audio_frames)

        min_effective_num_of_frames = video_md.min_effective_num_of_frames
        assert min_effective_num_of_frames >= self.sync_len

        seq_start_index = 0
        seq_end_index = min_effective_num_of_frames
        num_of_input_video_frames = seq_end_index - seq_start_index

        anchors = anchors[seq_start_index:(seq_end_index + self.ef_frames_window_size - 1)]
        audio_features = audio_features[:,
                         (seq_start_index * self.audio_video_ratio):
                         ((
                                  seq_end_index + self.audio_window_size // self.audio_video_ratio - 1) * self.audio_video_ratio)]

        assert len(anchors) == (audio_features.shape[1] // self.audio_video_ratio)
        assert num_of_input_video_frames >= self.sync_len
        assert len(anchors) >= self.sync_len + self.ef_frames_window_size - 1

        input_frames = []

        # augmentation is executed one by one, in order to use deterministic augmentation (otherwise you will get a random aug for each image (imgaug library))
        for i_frame in range(0, len(anchors)):
            seq_frame_index = seq_start_index + i_frame
            frame_path = '{}/{:0>5d}.jpg'.format(video_md.video_frames_path, (seq_frame_index + 1))

            with Image.open(frame_path) as pi:
                frame = np.array(pi)

            mouth = extract_mouth_from_frame(frame, anchors[i_frame], self.mouth_height, self.mouth_width)
            mouth = np.expand_dims(mouth, axis=0)
            mouth = self.rgb2gray_augmentor.augment_images(mouth)
            mouth = mouth[0, :, :, 0]

            md = FrameMetadata()
            md.data = mouth
            md.index = i_frame
            input_frames.append(md)

        input_frames = np.array(input_frames)
        assert len(input_frames) == len(anchors)

        tmp_input_frames = []
        for i in range(0, len(anchors) - self.ef_frames_window_size + 1):
            tmp_input_frames.append([x.data for x in input_frames[i:(i + self.ef_frames_window_size)]])

        input_frames = np.stack(tmp_input_frames)

        tmp_audio_features = []
        for i_video_frame in range(0, len(input_frames)):
            audio_index = i_video_frame * self.audio_video_ratio
            tmp_audio_features.append(audio_features[:, audio_index:(audio_index + self.audio_window_size)])

        input_audio_features = np.stack(tmp_audio_features)

        input_dict = {'visual_input': input_frames,
                      'audio_features': input_audio_features}

        return input_dict

    def load_one_train_set(self, video_index):
        assert self.db_folder_name == 'lrs2'

        video_md = self.videos_md[video_index]
        orig_number_of_frames_in_video = video_md.number_of_frames_in_video

        anchors = np.load(video_md.anchors_path)
        assert len(anchors) == orig_number_of_frames_in_video

        audio_features_path = '{}{}/{}_af.npy'.format(self.base_path, video_md.folder_name, video_md.file_name)
        audio_features = np.load(audio_features_path)

        orig_number_of_audio_frames = video_md.number_of_audio_frames
        assert audio_features.shape == (13, orig_number_of_audio_frames)

        min_effective_num_of_frames = video_md.min_effective_num_of_frames
        assert min_effective_num_of_frames >= self.sync_len

        seq_start_index = 0
        seq_end_index = min_effective_num_of_frames

        if min_effective_num_of_frames > self.max_seq_len:
            min_seq_start_index = 0
            max_seq_start_index = min_effective_num_of_frames - self.max_seq_len + 1
            seq_start_index = np.random.randint(min_seq_start_index, max_seq_start_index)
            seq_end_index = seq_start_index + self.max_seq_len

        num_of_input_video_frames = seq_end_index - seq_start_index

        anchors = anchors[seq_start_index:(seq_end_index + self.ef_frames_window_size - 1)]
        audio_features = audio_features[:,
                         (seq_start_index * self.audio_video_ratio):
                         ((
                                      seq_end_index + self.audio_window_size // self.audio_video_ratio - 1) * self.audio_video_ratio)]

        assert len(anchors) == (audio_features.shape[1] // self.audio_video_ratio)
        assert num_of_input_video_frames >= self.sync_len

        frames_actions = []
        duplications = []
        tmp_current_shift = 0
        num_of_frames_after_actions = 0

        current_number_of_diff_frames = len(anchors)

        for i_frame in range(0, len(anchors)):
            dup_can_be_added = tmp_current_shift < self.frames_shift_window_train
            drop_can_be_added = (tmp_current_shift == self.frames_shift_window_train) or np.abs(tmp_current_shift) < self.frames_shift_window_train

            drop_can_be_added = drop_can_be_added and (current_number_of_diff_frames > self.sync_len + self.ef_frames_window_size - 1)

            if (not dup_can_be_added and not drop_can_be_added) or np.random.rand() >= 0.5:
                frames_actions.append(FrameAction.Keep)
                duplications.append(0)
                num_of_frames_after_actions += 1
            elif (not dup_can_be_added) or (drop_can_be_added and np.random.rand() >= 0.5):
                frames_actions.append(FrameAction.Drop)
                duplications.append(0)
                tmp_current_shift -= 1
                current_number_of_diff_frames -= 1
            else:
                frames_actions.append(FrameAction.Duplicate)

                max_dup = self.max_frame_duplications
                if tmp_current_shift > 0:
                    max_dup = min(max_dup, self.frames_shift_window_train - tmp_current_shift + 1)

                current_dup = np.random.randint(1, max_dup)
                tmp_current_shift += current_dup
                duplications.append(current_dup)
                num_of_frames_after_actions += 1 + current_dup

        assert num_of_frames_after_actions >= self.sync_len + self.ef_frames_window_size - 1

        affine_augmentor = self.affine_augmentor.to_deterministic()
        image_channels_augmentor = self.image_channels_augmentor.to_deterministic()

        input_frames = []

        # augmentation is executed one by one, in order to use deterministic augmentation (otherwise you will get a random aug for each image (imgaug library))
        for i_action in range(0, len(frames_actions)):
            if frames_actions[i_action] == FrameAction.Keep or frames_actions[i_action] == FrameAction.Duplicate:
                seq_frame_index = seq_start_index + i_action
                frame_path = '{}/{:0>5d}.jpg'.format(video_md.video_frames_path, (seq_frame_index + 1))

                with Image.open(frame_path) as pi:
                    frame = np.array(pi)

                mouth = extract_mouth_from_frame(frame, anchors[i_action], self.mouth_height, self.mouth_width)
                mouth = np.expand_dims(mouth, axis=0)

                if not self.gen_mode:
                    mouth = affine_augmentor.augment_images(mouth)
                    mouth = image_channels_augmentor.augment_images(mouth)
                    mouth = self.image_dropout.augment_images(mouth)

                mouth = self.rgb2gray_augmentor.augment_images(mouth)
                mouth = mouth[0, :, :, 0]

                md = FrameMetadata()
                md.data = mouth
                md.index = i_action
                input_frames.append(md)

            if frames_actions[i_action] == FrameAction.Duplicate:
                for i in range(0, duplications[i_action]):
                    input_frames.append(md)

        input_frames = np.array(input_frames)
        assert len(input_frames) == num_of_frames_after_actions

        prediction = []
        input_frames_index = 0
        orig_frame_index = 0

        while orig_frame_index < len(anchors):
            if input_frames_index == len(input_frames):
                pred = input_frames_index - 1
                prediction.append(pred)
                # self.assert_prediction(input_frames[pred], all_frames[orig_frame_index], frames_actions[orig_frame_index])
                orig_frame_index += 1
            elif input_frames[input_frames_index].index > orig_frame_index:
                if input_frames_index == 0:
                    pred = input_frames_index
                    prediction.append(pred)
                    # self.assert_prediction(input_frames[pred], all_frames[orig_frame_index],
                    #                        frames_actions[orig_frame_index])
                else:
                    assert (orig_frame_index - input_frames[input_frames_index - 1].index) >= 0

                    if (orig_frame_index - input_frames[input_frames_index - 1].index) < (input_frames[input_frames_index].index - orig_frame_index):
                        pred = input_frames_index - 1
                        prediction.append(pred)
                        # self.assert_prediction(input_frames[pred], all_frames[orig_frame_index],
                        #                        frames_actions[orig_frame_index])
                        input_frames_index -= 1
                    else:
                        pred = input_frames_index
                        prediction.append(pred)
                        # self.assert_prediction(input_frames[pred], all_frames[orig_frame_index],
                        #                        frames_actions[orig_frame_index])

                orig_frame_index += 1
            elif input_frames[input_frames_index].index == orig_frame_index:
                while (input_frames_index + 1 < len(input_frames)) and input_frames[
                        input_frames_index + 1].index == orig_frame_index:
                    input_frames_index += 1
                pred = input_frames_index
                prediction.append(pred)
                # self.assert_prediction(input_frames[pred], all_frames[orig_frame_index],
                #                        frames_actions[orig_frame_index])
                orig_frame_index += 1
            elif input_frames[input_frames_index].index < orig_frame_index:
                input_frames_index += 1

        prediction = np.stack(prediction)

        max_diff = max(np.abs(np.arange(len(prediction)) - prediction))
        assert max_diff <= self.frames_shift_window_train

        tmp_input_frames = []
        for i in range(0, num_of_frames_after_actions - self.ef_frames_window_size + 1):
            tmp_input_frames.append([x.data for x in input_frames[i:(i + self.ef_frames_window_size)]])

        input_frames = np.stack(tmp_input_frames)

        if len(input_frames) > self.max_seq_len:
            input_frames = input_frames[:self.max_seq_len]

        out_of_range_predictions = np.where(prediction >= len(input_frames))[0]
        if len(out_of_range_predictions != 0):
            trim_index = out_of_range_predictions[0]
            prediction = prediction[:trim_index]
            assert len(prediction) >= self.sync_len

        max_shift_index = min(num_of_input_video_frames, len(prediction)) - self.sync_len + 1

        if max_shift_index > 0:
            shift = np.random.randint(0, max_shift_index)
        else:
            shift = 0

        tmp_audio_features = []
        for i_video_frame in range(shift, shift + self.sync_len):
            audio_index = i_video_frame * self.audio_video_ratio
            tmp_audio_features.append(audio_features[:, audio_index:(audio_index + self.audio_window_size)])

        input_audio_features = np.stack(tmp_audio_features)

        audio_seq_len = self.sync_len
        assert len(input_audio_features) == audio_seq_len

        video_seq_len = len(input_frames)

        target_prediction = prediction[shift:(shift + audio_seq_len)]
        assert target_prediction.max() < video_seq_len
        assert len(target_prediction) == self.sync_len

        if video_seq_len < self.max_seq_len:
            pad_size = self.max_seq_len - video_seq_len
            pad = np.zeros(pad_size * input_frames.shape[1] * input_frames.shape[2] * input_frames.shape[3]).reshape(
                (pad_size, *input_frames.shape[1:]))
            input_frames = np.concatenate((input_frames, pad))

        assert len(input_frames) == self.max_seq_len

        if not self.gen_mode:
            target_prediction = torch.tensor(target_prediction, dtype=torch.long)
            input_audio_features = torch.tensor(input_audio_features, dtype=torch.float)
            input_frames = torch.tensor(input_frames, dtype=torch.float)
            video_seq_len = torch.tensor(video_seq_len, dtype=torch.float)
            audio_seq_len = torch.tensor(audio_seq_len, dtype=torch.float)

        input_dict = {'visual_input': input_frames,
                      'target_prediction': target_prediction,
                      'audio_features': input_audio_features,
                      'video_seq_len': video_seq_len,
                      'audio_seq_len': audio_seq_len}

        return input_dict

    def load_one_test_set_single(self, video_index):
        assert self.db_folder_name == 'lrs2'

        video_md = self.videos_md[video_index]
        orig_number_of_frames_in_video = video_md.number_of_frames_in_video

        anchors = np.load(video_md.anchors_path)
        assert len(anchors) == orig_number_of_frames_in_video

        audio_features_path = '{}{}/{}_af.npy'.format(self.base_path, video_md.folder_name, video_md.file_name)
        audio_features = np.load(audio_features_path)

        orig_number_of_audio_frames = video_md.number_of_audio_frames
        assert audio_features.shape == (13, orig_number_of_audio_frames)

        min_effective_num_of_frames = video_md.min_effective_num_of_frames
        assert min_effective_num_of_frames >= self.sync_len

        seq_start_index = 0
        seq_end_index = min_effective_num_of_frames

        if min_effective_num_of_frames > self.max_seq_len:
            seq_end_index = seq_start_index + self.max_seq_len

        num_of_input_video_frames = seq_end_index - seq_start_index

        anchors = anchors[seq_start_index:(seq_end_index + self.ef_frames_window_size - 1)]
        audio_features = audio_features[:,
                         (seq_start_index * self.audio_video_ratio):
                         ((
                                      seq_end_index + self.audio_window_size // self.audio_video_ratio - 1) * self.audio_video_ratio)]

        assert len(anchors) == (audio_features.shape[1] // self.audio_video_ratio)

        max_shift_index = num_of_input_video_frames - self.sync_len + 1

        shift = video_md.test_shift

        shift = abs(shift)
        if abs(shift) >= max_shift_index:
            if shift != 0:
                sign = np.sign(shift)
            else:
                sign = 1

            shift = sign * (max_shift_index - 1)

        assert shift < max_shift_index

        # anchors = anchors[shift:(shift + self.sync_len + self.ef_frames_window_size - 1)]
        # assert len(anchors) == (self.sync_len + self.ef_frames_window_size - 1)

        audio_features = audio_features[:, (shift * self.audio_video_ratio):((
                                                                                         shift + self.sync_len + self.audio_window_size // self.audio_video_ratio - 1) * self.audio_video_ratio)]
        assert audio_features.shape[1] // self.audio_video_ratio == (
                    self.sync_len + self.audio_window_size // self.audio_video_ratio - 1)

        input_frames = []

        # augmentation is executed one by one, in order to use deterministic augmentation (otherwise you will get a random aug for each image (imgaug library))
        for i_frame in range(0, len(anchors)):
            seq_frame_index = seq_start_index + i_frame
            frame_path = '{}/{:0>5d}.jpg'.format(video_md.video_frames_path, (seq_frame_index + 1))

            with Image.open(frame_path) as pi:
                frame = np.array(pi)

            mouth = extract_mouth_from_frame(frame, anchors[i_frame], self.mouth_height, self.mouth_width)
            mouth = np.expand_dims(mouth, axis=0)
            mouth = self.rgb2gray_augmentor.augment_images(mouth)
            mouth = mouth[0, :, :, 0]
            input_frames.append(mouth)

        input_frames = np.array(input_frames)

        prediction = np.arange(shift, shift + self.sync_len)

        tmp_input_frames = []
        for i in range(0, num_of_input_video_frames):
            tmp_input_frames.append([x for x in input_frames[i:(i + self.ef_frames_window_size)]])

        input_frames = np.stack(tmp_input_frames)

        # for i in range(0, len(input_frames)):
        #     cv2.imwrite('c://temp//{}.png'.format(i), input_frames[i])

        tmp_audio_features = []
        for i_video_frame in range(0, self.sync_len):
            audio_index = i_video_frame * self.audio_video_ratio
            tmp_audio_features.append(audio_features[:, audio_index:(audio_index + self.audio_window_size)])

        input_audio_features = np.stack(tmp_audio_features)

        audio_seq_len = self.sync_len
        assert len(input_audio_features) == audio_seq_len
        # input_audio_features = np.flip(input_audio_features, 0).copy()

        video_seq_len = num_of_input_video_frames
        assert len(input_frames) == video_seq_len
        # input_frames = np.flip(input_frames, 0).copy()

        if video_seq_len < self.max_seq_len:
            pad_size = self.max_seq_len - video_seq_len
            pad = np.zeros(pad_size * input_frames.shape[1] * input_frames.shape[2] * input_frames.shape[3]).reshape(
                (pad_size, *input_frames.shape[1:]))
            input_frames = np.concatenate((input_frames, pad))

        assert len(input_frames) == self.max_seq_len

        target_prediction = torch.tensor(prediction, dtype=torch.long)
        input_audio_features = torch.tensor(input_audio_features, dtype=torch.float)
        input_frames = torch.tensor(input_frames, dtype=torch.float)
        video_seq_len = torch.tensor(video_seq_len, dtype=torch.float)
        audio_seq_len = torch.tensor(audio_seq_len, dtype=torch.float)

        input_dict = {'visual_input': input_frames,
                      'target_prediction': target_prediction,
                      'audio_features': input_audio_features,
                      'video_seq_len': video_seq_len,
                      'audio_seq_len': audio_seq_len,
                      'shift': shift,
                      'folder_name': video_md.folder_name,
                      'file_name': video_md.file_name}

        return input_dict

    def load_one_train_set_single(self, video_index):
        assert self.db_folder_name == 'lrs2'

        video_md = self.videos_md[video_index]
        orig_number_of_frames_in_video = video_md.number_of_frames_in_video

        anchors = np.load(video_md.anchors_path)
        assert len(anchors) == orig_number_of_frames_in_video

        audio_features_path = '{}{}/{}_af.npy'.format(self.base_path, video_md.folder_name, video_md.file_name)
        audio_features = np.load(audio_features_path)

        orig_number_of_audio_frames = video_md.number_of_audio_frames
        assert audio_features.shape == (13, orig_number_of_audio_frames)

        min_effective_num_of_frames = video_md.min_effective_num_of_frames
        assert min_effective_num_of_frames >= self.sync_len

        seq_start_index = 0
        seq_end_index = min_effective_num_of_frames

        if min_effective_num_of_frames > self.max_seq_len:
            min_seq_start_index = 0
            max_seq_start_index = min_effective_num_of_frames - self.max_seq_len + 1
            seq_start_index = np.random.randint(min_seq_start_index, max_seq_start_index)
            seq_end_index = seq_start_index + self.max_seq_len

        num_of_input_video_frames = seq_end_index - seq_start_index

        anchors = anchors[seq_start_index:(seq_end_index+self.ef_frames_window_size-1)]
        audio_features = audio_features[:,
                         (seq_start_index * self.audio_video_ratio):
                         ((seq_end_index + self.audio_window_size // self.audio_video_ratio - 1) * self.audio_video_ratio)]

        assert len(anchors) == (audio_features.shape[1] // self.audio_video_ratio)

        max_shift_index = num_of_input_video_frames - self.sync_len + 1
        shift = np.random.randint(0, max_shift_index)

        # anchors = anchors[shift:(shift + self.sync_len + self.ef_frames_window_size - 1)]
        # assert len(anchors) == (self.sync_len + self.ef_frames_window_size - 1)

        audio_features = audio_features[:, (shift * self.audio_video_ratio):((shift + self.sync_len + self.audio_window_size // self.audio_video_ratio - 1) * self.audio_video_ratio)]
        assert audio_features.shape[1] // self.audio_video_ratio == (self.sync_len + self.audio_window_size // self.audio_video_ratio - 1)

        affine_augmentor = self.affine_augmentor.to_deterministic()
        image_channels_augmentor = self.image_channels_augmentor.to_deterministic()

        input_frames = []

        # augmentation is executed one by one, in order to use deterministic augmentation (otherwise you will get a random aug for each image (imgaug library))
        for i_frame in range(0, len(anchors)):
            seq_frame_index = seq_start_index + i_frame
            frame_path = '{}/{:0>5d}.jpg'.format(video_md.video_frames_path, (seq_frame_index + 1))

            with Image.open(frame_path) as pi:
                frame = np.array(pi)

            mouth = extract_mouth_from_frame(frame, anchors[i_frame], self.mouth_height, self.mouth_width)
            mouth = np.expand_dims(mouth, axis=0)
            mouth = affine_augmentor.augment_images(mouth)
            mouth = image_channels_augmentor.augment_images(mouth)
            mouth = self.image_dropout.augment_images(mouth)
            mouth = self.rgb2gray_augmentor.augment_images(mouth)
            mouth = mouth[0, :, :, 0]
            input_frames.append(mouth)

        input_frames = np.array(input_frames)

        prediction = np.arange(shift, shift + self.sync_len)

        tmp_input_frames = []
        for i in range(0, num_of_input_video_frames):
            tmp_input_frames.append([x for x in input_frames[i:(i + self.ef_frames_window_size)]])

        input_frames = np.stack(tmp_input_frames)

        # for i in range(0, len(input_frames)):
        #     cv2.imwrite('c://temp//{}.png'.format(i), input_frames[i])

        tmp_audio_features = []
        for i_video_frame in range(0, self.sync_len):
            audio_index = i_video_frame * self.audio_video_ratio
            tmp_audio_features.append(audio_features[:, audio_index:(audio_index + self.audio_window_size)])

        input_audio_features = np.stack(tmp_audio_features)

        audio_seq_len = self.sync_len
        assert len(input_audio_features) == audio_seq_len
        # input_audio_features = np.flip(input_audio_features, 0).copy()

        video_seq_len = num_of_input_video_frames
        assert len(input_frames) == video_seq_len
        # input_frames = np.flip(input_frames, 0).copy()

        if video_seq_len < self.max_seq_len:
            pad_size = self.max_seq_len - video_seq_len
            pad = np.zeros(pad_size * input_frames.shape[1] * input_frames.shape[2] * input_frames.shape[3]).reshape((pad_size, *input_frames.shape[1:]))
            input_frames = np.concatenate((input_frames, pad))

        assert len(input_frames) == self.max_seq_len

        target_prediction = torch.tensor(prediction, dtype=torch.long)
        input_audio_features = torch.tensor(input_audio_features, dtype=torch.float)
        input_frames = torch.tensor(input_frames, dtype=torch.float)
        video_seq_len = torch.tensor(video_seq_len, dtype=torch.float)
        audio_seq_len = torch.tensor(audio_seq_len, dtype=torch.float)

        input_dict = {'visual_input': input_frames,
                      'target_prediction': target_prediction,
                      'audio_features': input_audio_features,
                      'video_seq_len': video_seq_len,
                      'audio_seq_len': audio_seq_len}

        return input_dict
