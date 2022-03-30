#!/usr/bin/env python

from __future__ import print_function
import sys
from common import *
from Model.model import Model
from Model.dataset import *
import time
from collections import Counter


class SingleShiftEvaluator:
    def __init__(self, db_type, config, db_config):
        super(SingleShiftEvaluator, self).__init__()
        self.config = config
        self.db_config = db_config
        self.db_loader, self.dataset = init_dataset(self.config, self.db_config, db_type, True)
        self.dataset_size = len(self.dataset)

        self.gpu_ids = []
        if config.general.gpu_ids != '':
            self.gpu_ids = np.array(self.config.general.gpu_ids.split(' ')).astype(np.int)
        if torch.cuda.is_available() and len(self.gpu_ids) != 0:
            self.device = torch.device('cuda:{0}'.format(self.gpu_ids[0]))
        else:
            self.device = torch.device('cpu')

    def eval(self, model):
        start = time.time()

        print('evaluating...')

        model.eval()

        with torch.no_grad():
            perfects = []
            frames_errors = []
            shifts = []

            for i, video_data in enumerate(self.db_loader):
                visual_input = video_data['visual_input'].to(self.device)
                target_prediction = video_data['target_prediction'].to(self.device)
                audio_features = video_data['audio_features'].to(self.device)
                video_seq_len = video_data['video_seq_len'].to(self.device)
                audio_seq_len = video_data['audio_seq_len'].to(self.device)
                current_shifts = video_data['shift']

                _, per_count, f_errors, _, _ = model(visual_input, target_prediction, audio_features, video_seq_len,
                                                  audio_seq_len, True)

                perfects.append(per_count.cpu().numpy())
                frames_errors.append(f_errors.cpu().numpy())
                shifts.append(current_shifts.sum() / len(current_shifts))

            shifts = np.array(shifts)
            avg_shifts = np.average(shifts)

            perfects = np.array(perfects)
            avg_perfects = np.average(perfects)

            frames_errors = np.array(frames_errors)
            avg_frames_error = np.average(frames_errors)

            print('avg_perfects = {}'.format(avg_perfects))
            print('avg_frames_error = {}'.format(avg_frames_error))
            print('avg_shifts = {}'.format(avg_shifts))

        model.train()
        end = time.time()

        print('evaluation took {}'.format(end - start))


def run_once(evaluator):
    model = Model(evaluator.config)
    # model = torch.nn.DataParallel(model)
    model.to(evaluator.device)
    evaluator.eval(model)


def run_seq(evaluator):
    for i in range(0, 5):
        print(i)
        model = Model(evaluator.config, epoch=str(i))
        # model = torch.nn.DataParallel(model)
        model.to(evaluator.device)
        evaluator.eval(model)


def main(argv):
    os.chdir('..')
    config = ConfigParser("Model/config.yaml")
    db_config = ConfigParser("dbs_config.yaml")

    db_name = db_config.general.db_name
    assert db_name == 'lrs2_dataset'

    evaluator = SingleShiftEvaluator(DBType.Validation, config, db_config)
    # multiprocessing.set_start_method('spawn', force=True)
    run_once(evaluator)
    #run_seq(evaluator)


if __name__ == '__main__':
    main(sys.argv)
