import os
from Model.networks import *
import torch
import random
import numpy as np
from common import *


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
    def __init__(self, config, epoch='latest', syncnet_model_path=None):
        super(Model, self).__init__()

        self.config = config
        self.checkpoints_path = '{0}/{1}/'.format(self.config.general.output_path,
                                                  self.config.general.checkpoints_folder)

        self.syncnet = SyncNet(config)
        self.syncnet.apply(weights_init)
        self.try_load_network(self.syncnet, 'syncnet', epoch, syncnet_model_path)
        self.SOS_token = config.general.max_seq_len
        self.padding_value = config.general.max_seq_len + 1
        self.sync_len = self.config.general.sync_len
        self.max_seq_len = self.config.general.max_seq_len

        gpu_ids = []
        if config.general.gpu_ids != '':
            gpu_ids = np.array(config.general.gpu_ids.split(' ')).astype(np.int)
        if torch.cuda.is_available() and len(gpu_ids) != 0:
            self.device = torch.device('cuda:{0}'.format(gpu_ids[0]))
        else:
            self.device = torch.device('cpu')

        self.syncnet_classifier = SyncNetClassifier(config)
        self.syncnet_classifier.apply(weights_init)
        self.try_load_network(self.syncnet_classifier, 'syncnet_classifier', epoch)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.padding_value)
        self.contrastive_loss = ContrastiveLoss(config, margin=20)

        self.teacher_forcing_ratio = config.rnn_decoder.teacher_forcing_ratio

        self.current_lr = self.config.train.lr
        params = list(self.syncnet.parameters()) + list(self.syncnet_classifier.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.current_lr, betas=(0.5, 0.999))

    def try_load_network(self, network, network_label, epoch_label, path=None):
        if path is None:
            file_path = '{0}{1}_{2}.dat'.format(self.checkpoints_path, epoch_label, network_label)
        else:
            file_path = path

        if os.path.isfile(file_path):
            if torch.cuda.is_available():
                #network.load_state_dict(torch.load(file_path))
                network.load_state_dict(torch.load(file_path, map_location={'cuda:1': 'cuda:0'}))
                #network.load_state_dict(torch.load(file_path, map_location={'cuda:0': 'cuda:1'}))
            else:
                network.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
            print('{} was loaded'.format(file_path))

    def encode(self, visual_input, audio_features, video_seq_len, audio_seq_len, get_all_hidden_states=False):
        num_of_audio_frames = audio_features.shape[1]
        batch_size = visual_input.shape[0]

        visual_encoded, audio_encoded = self.get_embeddings(visual_input, audio_features)

        l2_features = []
        for i_batch in range(0, batch_size):
            current_visual = visual_encoded[i_batch]
            for i_audio in range(0, num_of_audio_frames):
                current_audio = audio_encoded[i_batch, i_audio].unsqueeze(0).expand(visual_encoded.shape[1], -1)
                current_features = ContrastiveLoss.calc_dist(current_audio, current_visual)
                l2_features.append(current_features)

        l2_features = torch.stack(l2_features)
        l2_features = l2_features.view(batch_size, num_of_audio_frames, -1)

        encoded_sequence, encoder_hidden, encoder_cell = self.syncnet_classifier.encode(l2_features, get_all_hidden_states)

        return encoded_sequence, encoder_hidden, encoder_cell

    def get_embeddings(self, visual_input, audio_features):
        batch_size = visual_input.shape[0]
        num_of_visual_frames = visual_input.shape[1]
        num_of_audio_frames = audio_features.shape[1]

        visual_input = visual_input.view(batch_size * num_of_visual_frames, 1, *visual_input.shape[-3:])
        visual_encoded = self.syncnet.forward_visual(visual_input)
        visual_encoded = visual_encoded.view(batch_size, num_of_visual_frames, -1)

        audio_features = audio_features.view(batch_size * num_of_audio_frames, 1, *audio_features.shape[-2:])
        audio_encoded = self.syncnet.forward_audio(audio_features)
        audio_encoded = audio_encoded.view(batch_size, num_of_audio_frames, -1)

        return visual_encoded, audio_encoded

    def run_forward(self, visual_input, target_prediction, audio_features, video_seq_len, audio_seq_len, is_test=False, initial_output=None):
        encoded_sequence, decoder_hidden, _ = self.encode(visual_input, audio_features, video_seq_len, audio_seq_len)

        batch_size = visual_input.shape[0]

        if initial_output is None:
            initial_output = self.SOS_token
        else:
            assert batch_size == 1

        decoder_output = torch.full((batch_size, 1), initial_output, device=self.device, dtype=torch.long)[:, 0]
        attn_context = torch.zeros((batch_size, 512), device=self.device)
        decoder_cell = torch.zeros_like(decoder_hidden, device=self.device)
        final_outputs = torch.zeros((self.sync_len, batch_size, self.max_seq_len), device=self.device)

        for di in range(self.sync_len):
            decoder_output, decoder_hidden, decoder_cell, attn_context = self.syncnet_classifier.decode(
                decoder_output, decoder_hidden, decoder_cell, attn_context, encoded_sequence)

            final_outputs[di] = decoder_output

            use_teacher_forcing = True if (random.random() < self.teacher_forcing_ratio and not is_test) else False

            topi = decoder_output.topk(1)[1][:, 0]
            if use_teacher_forcing:
                decoder_output = target_prediction[:, di].detach()
            else:
                decoder_output = topi.detach()


        final_outputs = final_outputs.permute(1, 0, 2)
        final_outputs = final_outputs.contiguous().view(-1, final_outputs.shape[-1])

        return final_outputs

    def forward(self, visual_input, target_prediction, audio_features, video_seq_len, audio_seq_len, is_test=False):
        final_outputs = self.run_forward(visual_input, target_prediction, audio_features, video_seq_len, audio_seq_len, is_test)
        target = target_prediction[:, :final_outputs.shape[0]].contiguous().view(-1)

        loss_output = self.ce_loss(final_outputs, target)

        if is_test:
            predictions = torch.max(final_outputs, dim=1)[1]
            perfects = (predictions[target != self.padding_value] == target[target != self.padding_value])
            perfects = perfects.float().sum() / len(perfects)

            frames_errors = torch.abs(predictions[target != self.padding_value] - target[target != self.padding_value])
            frames_errors = frames_errors.float().sum() / len(frames_errors)

            return loss_output, perfects, frames_errors, final_outputs, target
        else:
            return loss_output, final_outputs, target

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
        self.save_network(self.syncnet_classifier, 'syncnet_classifier', which_epoch)

    def beam_search(self, visual_input, target_prediction, audio_features, video_seq_len, audio_seq_len, beam_size=10):
        # taken from:
        # https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/tools/beam_search.py

        encoded_sequence, decoder_hidden, _ = self.encode(visual_input, audio_features, video_seq_len, audio_seq_len)

        max_seq_len = max(audio_seq_len.cpu().numpy()).astype(np.int)
        batch_size = visual_input.shape[0]
        decoder_output = torch.full((batch_size, 1), self.SOS_token, device=self.device, dtype=torch.long)[:, 0]
        attn_context = torch.zeros((batch_size, 512), device=self.device)
        decoder_cell = torch.zeros_like(decoder_hidden, device=self.device)

        partial_sequences = [TopN(beam_size) for _ in range(batch_size)]

        decoder_output, decoder_hidden, decoder_cell, attn_context = self.syncnet_classifier.decode(
            decoder_output, decoder_hidden, decoder_cell, attn_context, encoded_sequence)

        log_softmax = nn.LogSoftmax(1)
        logprobs = log_softmax(decoder_output)
        logprobs, words = logprobs.topk(beam_size, 1)
        words = words.detach()
        logprobs = logprobs.detach()
        decoder_output = words

        for b in range(batch_size):
            for k in range(beam_size):
                seq = Sequence(
                    output=[self.SOS_token] + [words[b][k]],
                    decoder_output=decoder_output[b][k],
                    decoder_hidden=decoder_hidden[:, b],
                    decoder_cell=decoder_cell[:, b],
                    attn_context=attn_context[b],
                    encoded_sequence=encoded_sequence[:, b],
                    logprob=logprobs[b][k],
                    score=logprobs[b][k])
                partial_sequences[b].push(seq)

        for _ in range(max_seq_len-1):
            partial_sequences_list = [p.extract() for p in partial_sequences]
            for p in partial_sequences:
                p.reset()

            flattened_partial = [s for sub_partial in partial_sequences_list for s in sub_partial]

            decoder_output = torch.stack([c.decoder_output for c in flattened_partial])
            decoder_hidden = torch.stack([c.decoder_hidden for c in flattened_partial], dim=1)
            decoder_cell = torch.stack([c.decoder_cell for c in flattened_partial], dim=1)
            attn_context = torch.stack([c.attn_context for c in flattened_partial])
            encoded_sequence = torch.stack([c.encoded_sequence for c in flattened_partial], dim=1)

            decoder_output, decoder_hidden, decoder_cell, attn_context = self.syncnet_classifier.decode(
                decoder_output, decoder_hidden, decoder_cell, attn_context, encoded_sequence)

            logprobs = log_softmax(decoder_output)
            logprobs, words = logprobs.topk(beam_size, 1)
            words = words.detach()
            logprobs = logprobs.detach()
            decoder_output = words

            idx = 0
            for b in range(batch_size):
                for partial in partial_sequences_list[b]:
                    k = 0
                    while k < beam_size and ((len(partial.output)-1) < video_seq_len[b].cpu().numpy()):
                        w = words[idx][k]
                        output = partial.output + [w]
                        logprob = partial.logprob + logprobs[idx][k]
                        score = logprob

                        beam = Sequence(
                            output=output,
                            decoder_output=decoder_output[idx][k],
                            decoder_hidden=decoder_hidden[:, idx],
                            decoder_cell=decoder_cell[:, idx],
                            attn_context=attn_context[idx],
                            encoded_sequence=encoded_sequence[:, idx],
                            logprob=logprob,
                            score=score)

                        current_seq_len = video_seq_len[b].cpu().numpy()
                        if (len(beam.output)-1) == current_seq_len:
                            pad_size = int(max_seq_len - current_seq_len)
                            if pad_size != 0:
                                beam.output = beam.output + list(torch.tensor(np.full(pad_size, self.padding_value), device=self.device, dtype=torch.long))

                        partial_sequences[b].push(beam)
                        k += 1
                    idx += 1

            for b in range(batch_size):
                if not partial_sequences[b].size():
                    for x in partial_sequences_list[b]:
                        partial_sequences[b].push(x)
        seqs = [complete.extract(sort=True)[0]
                for complete in partial_sequences]

        target = target_prediction[:, :max_seq_len].contiguous().view(-1)
        predictions = torch.stack([torch.stack(s.output[1:]) for s in seqs]).view(-1)

        perfects = (predictions[target != self.padding_value] == target[target != self.padding_value])
        perfects = perfects.float().sum() / len(perfects)

        frames_errors = torch.abs(predictions[target != self.padding_value] - target[target != self.padding_value])
        frames_errors = frames_errors.float().sum() / len(frames_errors)

        return perfects, frames_errors
