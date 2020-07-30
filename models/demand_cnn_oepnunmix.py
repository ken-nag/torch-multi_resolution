from torch.nn import LSTM, Linear, BatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNOpenUnmix(nn.Module):
    def __init__(self,cfg):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        n_fft=4096,
        n_hop=1024,
        hidden_size=512,
        sample_rate=44100,
        nb_layers=3,
        power=1,
        """
        super(CNNOpenUnmix, self).__init__()

        self.nb_output_bins = cfg['f_size']
        self.nb_bins = self.nb_output_bins

        self.hidden_size = cfg['hidden_size']

        self.fc1 = Linear(
            self.nb_bins, self.hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(self.hidden_size)

        lstm_hidden_size = self.hidden_size // 2

        self.lstm = LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=3,
            bidirectional=True,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=self.hidden_size*2,
            out_features=self.hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(self.hidden_size)

        self.fc3 = Linear(
            in_features=self.hidden_size,
            out_features=self.nb_output_bins,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins)

    def forward(self, x):
        x = x.permute(2,0,1)
        nb_frames, nb_samples, nb_bins = x.data.shape

        x = self.fc1(x.reshape(-1, self.nb_bins))
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = torch.tanh(x)

        lstm_out = self.lstm(x)

        x = torch.cat([x, lstm_out[0]], -1)

        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)

        x = x.reshape(nb_frames, nb_samples, self.nb_output_bins)
        x = x.permute(1,2,0)

        mask = F.relu(x)

        return mask