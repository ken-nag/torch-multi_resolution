from torch.nn import LSTM, Linear, BatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNOpenUnmix(nn.Module):
    def __init__(self,cfg):
        super(CNNOpenUnmix, self).__init__()
        self.f_size = cfg['f_size']
        
        self.kernel = cfg['kernel']
        self.stride = cfg['stride']
        self.channel = cfg['channel']
        self.dilation = cfg['dilation']
        
        self.mix_kernel=cfg['mix_kernel']
        self.mix_stride=cfg['mix_stride']
        self.mix_channel=cfg['mix_channel']
        self.mix_dilation=cfg['mix_dilation']
        
        self.hidden_size = cfg['hidden_size']
        self.first_linear_out = cfg['first_linear_out']
        self.leakiness = 0.2
        
        self.bathc_norm = nn.BatchNorm2d(1)
        self.encoder = self._encoder(channels=self.channel, 
                                     kernel_size=self.kernel, 
                                     stride=self.stride, 
                                     dilation=self.dilation)
        
        self.mix_encoder = self._encoder(channels=self.mix_channel, 
                                         kernel_size=self.mix_kernel, 
                                         stride=self.mix_stride, 
                                         dilation=self.mix_dilation)
        
        self.compressor = self._encoder(channels=(self.mix_channel[1],1), 
                                        kernel_size=(1,1), 
                                        stride=(1,1))
        
        self.first_linear_in = int(np.ceil(np.ceil(self.f_size/self.stride[0])/self.mix_stride[0]))
          
        self.nb_output_bins = cfg['f_size']
        self.nb_bins = self.nb_output_bins

        self.fc1 = Linear(
            self.first_linear_in, self.hidden_size,
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