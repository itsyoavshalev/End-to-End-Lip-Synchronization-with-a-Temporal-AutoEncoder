#!/usr/bin/env python

from __future__ import print_function
import sys
from common import *
from SyncNet.model import Model
from SyncNet.dataset import *
import time
from collections import Counter


class Evaluator:
    def __init__(self, db_type, config, db_config):
        super(Evaluator, self).__init__()
        self.config = config
        self.db_config = db_config
        self.db_loader, self.dataset = init_dataset(self.config, self.db_config, db_type)
        self.dataset_size = len(self.dataset)
        self.gpu_ids = np.array(self.config.general.gpu_ids.split(' ')).astype(np.int)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:{0}'.format(self.gpu_ids[0]))
        else:
            self.device = torch.device('cpu')

    def eval(self, model, max_iterations):
        start = time.time()

        print('evaluating...')

        model.eval()
        positive_losses = []
        negative_losses = []

        with torch.no_grad():
            visual_temporal_length = self.config.general.ef_frames_window_size

            if max_iterations == 0:
                max_iterations = self.dataset.get_number_of_videos()
            else:
                max_iterations = min(max_iterations, self.dataset.get_number_of_videos())

            perfects = []
            frames_errors = []
            for video_id in range(0, max_iterations):
                video_data = self.dataset.load_video_inputs(video_id)

                video_frames = video_data['video_frames']
                audio_features = video_data['audio_features']
                video_frames_shift = video_data['video_frames_shift']

                if video_frames_shift > 0:
                    audio_frames_shift = video_frames_shift * self.dataset.audio_video_ratio
                    audio_features = audio_features[:, audio_frames_shift:]
                else:
                    video_frames = video_frames[np.abs(video_frames_shift):]

                max_frames = len(video_frames) // visual_temporal_length * visual_temporal_length

                if max_frames <= 0:
                    continue

                video_frames = video_frames[:max_frames]
                video_frames = video_frames.view(-1, 1, visual_temporal_length, video_frames.shape[-2], video_frames.shape[-1])
                video_frames = video_frames.to(self.device)

                encoded_video_frames = model.syncnet.forward_visual(video_frames)

                chuncked_audio_features = []
                for i in range(0, audio_features.shape[1], self.dataset.audio_video_ratio):
                    start_index = i
                    end_index = start_index+self.config.general.audio_window_size

                    if end_index > audio_features.shape[1]:
                        break

                    current_audio_features = audio_features[:, start_index:end_index]
                    assert current_audio_features.shape == (13, self.config.general.audio_window_size)
                    chuncked_audio_features.append(current_audio_features)

                if len(chuncked_audio_features) <= 0:
                    continue
				
                audio_features = torch.stack(chuncked_audio_features)

                audio_features = audio_features.to(self.device).unsqueeze(1)
                encoded_audio_features = model.syncnet.forward_audio(audio_features)

                search_range_frames = self.config.general.frames_search_window_test

                tmp_predictions = []
                # each item in encoded_video_frames represents 5 video frames
                # each item in encoded_audio_features represents 20 audio features, which has the duration of 5 video frames
                # neighbor items in encoded_audio_features has a time gap of 1 video frame
                for i in range(0, len(encoded_video_frames)):
                    current_video_features = encoded_video_frames[i]
                    current_first_frame_index = i * visual_temporal_length
                    min_audio_index = max(0, current_first_frame_index - search_range_frames)
                    max_audio_index = min(current_first_frame_index + search_range_frames + 1, len(encoded_audio_features))
                    correct_audio_index = current_first_frame_index - video_frames_shift

                    dists = []
                    for j in range(min_audio_index, max_audio_index):
                        current_audio_features = encoded_audio_features[j]
                        dist = model.loss.calc_dist(current_video_features, current_audio_features)
                        true_pair = j == correct_audio_index

                        if true_pair:
                            y = torch.tensor(1, dtype=torch.float).to(self.device)
                        else:
                            y = torch.tensor(0, dtype=torch.float).to(self.device)

                        loss = model.loss(current_video_features.unsqueeze(0),
                                          current_audio_features.unsqueeze(0),
                                          y.unsqueeze(0))

                        if true_pair:
                            positive_losses.append(loss.item())
                        else:
                            negative_losses.append(loss.item())

                        dists.append(dist)

                    dists = np.array(dists)
                    predicted = current_first_frame_index - (min_audio_index + np.argmin(dists))
                    tmp_predictions.append(predicted)

                tmp_predictions = np.array(tmp_predictions)
                hist = Counter(tmp_predictions)
                voting_pred = hist.most_common(1)[0][0]
                perfect = int(voting_pred) == video_frames_shift
                perfects.append(perfect)
                frames_errors.append(np.abs(voting_pred - video_frames_shift))

            perfects = np.array(perfects)
            frames_errors = np.array(frames_errors)

            positive_losses = np.array(positive_losses)
            avg_positive_loss = np.average(positive_losses)

            negative_losses = np.array(negative_losses)
            avg_neg_loss = np.average(negative_losses)

            final_loss = (avg_positive_loss + avg_neg_loss) / 2
            avg_perfects = np.average(perfects)
            avg_frames_error = np.average(frames_errors)

            print('final_loss = {}'.format(final_loss))
            print('avg_positive_loss = {}'.format(avg_positive_loss))
            print('avg_neg_loss = {}'.format(avg_neg_loss))
            print('avg_perfects = {}'.format(avg_perfects))
            print('avg_frames_error = {}'.format(avg_frames_error))

        model.train()
        end = time.time()

        print('evaluation took {}'.format(end - start))
        return final_loss


def run_once(evaluator):
    model = Model(evaluator.config)
    # model = torch.nn.DataParallel(model)
    model.to(evaluator.device)
    evaluator.eval(model, max_iterations=0)


def run_seq(evaluator):
    for i in range(0, 22):
        print(i)
        model = Model(evaluator.config, str(i))
        # model = torch.nn.DataParallel(model)
        model.to(evaluator.device)
        evaluator.eval(model, max_iterations=0)


def main(argv):
    os.chdir('..')
    config = ConfigParser("SyncNet/config.yaml")
    db_config = ConfigParser("dbs_config.yaml")
    evaluator = Evaluator(DBType.Validation, config, db_config)
    # multiprocessing.set_start_method('spawn', force=True)
    run_once(evaluator)
    #run_seq(evaluator)


if __name__ == '__main__':
    main(sys.argv)
