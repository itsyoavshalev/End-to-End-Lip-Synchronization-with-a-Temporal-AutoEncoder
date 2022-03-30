import torch.nn as nn
import torch.nn.init as init
import torch
import torch.functional as F


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class SyncNet(nn.Module):
    def __init__(self, config, num_layers_in_fc_layers=1024):
        super(SyncNet, self).__init__()

        self.config = config

        self.audio_conv_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),

            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 2048, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )

        self.audio_fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, num_layers_in_fc_layers),
        )

        self.visual_fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, num_layers_in_fc_layers),
        )

        self.visual_conv_encoder = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(96, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(256, 2048, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True),
        )

    def forward_audio(self, audio_input):
        cnn_features = self.audio_conv_encoder(audio_input)
        cnn_features = cnn_features.view(cnn_features.shape[0], -1)
        output = self.audio_fc(cnn_features)

        return output

    def forward_visual(self, visual_input):
        cnn_features = self.visual_conv_encoder(visual_input)
        cnn_features = cnn_features.view(cnn_features.shape[0], -1)
        output = self.visual_fc(cnn_features)

        return output


class SyncNetClassifier(nn.Module):
    def __init__(self, config):
        super(SyncNetClassifier, self).__init__()

        # encoder
        self.rnn_encoder = nn.LSTM(config.general.max_seq_len, 512, 3)

        # decoder
        self.decoder_hidden_size = 512
        self.decoder_output_size = config.general.max_seq_len
        self.dropout_p = 0.1
        self.embedding = nn.Embedding(self.decoder_output_size + 2, self.decoder_hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn_decoder = nn.LSTM(self.decoder_hidden_size * 2, self.decoder_hidden_size, 3)

        # attention
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, 256)
        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(256, 1, bias=False)
        self.sm = nn.Softmax(0)

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, config.general.max_seq_len)
        )

    def encode(self, l2_features, get_all_hidden_states=False):
        x = l2_features.permute(1, 0, 2)
        outputs = []
        hiddens = []
        cells = []

        if get_all_hidden_states:
            c_i = None
            h_i = None
            for i in range(0, x.shape[0]):
                if h_i is not None and c_i is not None:
                    output_i, (h_i, c_i) = self.rnn_encoder(x[i:(i+1)], (h_i, c_i))
                else:
                    output_i, (h_i, c_i) = self.rnn_encoder(x[i:(i + 1)])
                outputs.append(output_i)
                hiddens.append(h_i)
                cells.append(c_i)

            outputs = torch.cat(outputs)
            hiddens = torch.stack(hiddens)
            cells = torch.stack(cells)

            return outputs, hiddens, cells
        else:
            outputs, (hn, cn) = self.rnn_encoder(x)
            return outputs, hn, cn

    def decode(self, prev_output, prev_hidden, prev_decoder_cell, prev_attn_context, encoder_outputs):
        embedded = self.embedding(prev_output)
        embedded = self.dropout(embedded)
        concatenated = torch.cat((embedded, prev_attn_context), dim=1)
        rnn_input = concatenated.unsqueeze(0)
        decoder_output, (decoder_hidden, decoder_cell) = self.rnn_decoder(rnn_input, (prev_hidden, prev_decoder_cell))
        decoder_output = decoder_output[0]

        ot = self.fc2(encoder_outputs)
        ht = self.fc1(decoder_hidden[-1]).unsqueeze(0)

        ht = ht.repeat(encoder_outputs.shape[0], 1, 1)

        out = ot + ht
        out = self.tanh(out)
        out = self.fc3(out)
        weights = self.sm(out)[:, :, 0]
        attn_context = weights.permute(1, 0).unsqueeze(1).bmm(encoder_outputs.permute(1, 0, 2))
        attn_context = attn_context[:, 0]

        mlp_input = torch.cat((decoder_output, attn_context), dim=1)
        mlp_output = self.mlp(mlp_input)

        return mlp_output, decoder_hidden, decoder_cell, attn_context
