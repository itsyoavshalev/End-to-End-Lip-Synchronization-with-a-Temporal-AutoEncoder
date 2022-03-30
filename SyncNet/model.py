import os
from SyncNet.networks import *
import numpy as np
import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, config, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.config = config

    @staticmethod
    def check_type_forward(in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.shape[0]

        return loss

    @staticmethod
    def calc_dist(x0, x1):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), -1)
        dist = torch.sqrt(dist_sq)

        return dist


class Model(torch.nn.Module):
    def __init__(self, config, epoch='latest'):
        super(Model, self).__init__()

        self.config = config
        self.checkpoints_path = '{0}/{1}/'.format(self.config.general.output_path, self.config.general.checkpoints_folder)

        self.syncnet = SyncNet(config)
        self.syncnet.apply(weights_init)
        self.try_load_network(self.syncnet, 'syncnet', epoch)

        self.loss = ContrastiveLoss(config, margin=20)

        self.current_lr = self.config.train.lr
        params = list(self.syncnet.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.current_lr, betas=(0.5, 0.999))

    def try_load_network(self, network, network_label, epoch_label):
        file_path = '{0}{1}_{2}.dat'.format(self.checkpoints_path, epoch_label, network_label)

        if os.path.isfile(file_path):
            if torch.cuda.is_available():
                network.load_state_dict(torch.load(file_path))
            else:
                network.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
            print('{} was loaded'.format(file_path))

    def forward(self, visual_input, target_prediction, audio_features):
        visual_outputs = self.syncnet.forward_visual(visual_input)
        audio_outputs = self.syncnet.forward_audio(audio_features)

        loss_output = self.loss(visual_outputs, audio_outputs, target_prediction)

        return loss_output

    def update_learning_rate(self, new_lr=None):
        if new_lr is None:
            new_lr = self.current_lr / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('learning rate was updated: %f -> %f' % (self.current_lr, new_lr))
        self.current_lr = new_lr

    def save_network(self, network, network_label, epoch_label):
        if not os.path.isdir(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)

        file_path = '{0}/{1}_{2}.dat'.format(self.checkpoints_path, epoch_label, network_label)
        torch.save(network.state_dict(), file_path)

    def save(self, which_epoch):
        self.save_network(self.syncnet, 'syncnet', which_epoch)
