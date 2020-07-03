import torch
import torch.nn as nn
import numpy as np

class FeatExtractorBlstm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.f_size = cfg['f_size']
        
        self.kernel = cfg['kernel']
        self.stride = cfg['stride']
        self.channel = cfg['channel']

        self.mix_kernel=cfg['mix_kernel']
        self.mix_stride=cfg['mix_stride']
        self.mix_channel=cfg['mix_channel']
        
        self.hidden_size = cfg['hidden_size']
        
        self.encoder = self._encoder(channels=self.channel, kernel_size=self.kernel, stride=self.stride)
        self.mix_encoder = self._encoder(channels=self.mix_channel, kernel_size=self.mix_stride, stride=self.mix_stride)
        self.compressor = self._encoder(channels=(self.mix_channel[1],1), kernel_size=(1,1), stride=(1,1))
        
        blstm_input_size = int(np.ceil(self.f_size/self.stride[0]))
        self.blstm_block = nn.LSTM(input_size=blstm_input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True, 
                                   batch_first=True)
        
        self.last_linear = nn.Linear(in_features=self.hidden_size*2, out_features=self.f_size)
    
        
    def _encoder(self, channels, kernel_size, stride):
        padding = self._kernel_pad(kernel_size)
        return nn.Sequential(nn.Conv2d(in_channels=channels[0],
                                       out_channels=channels[1],
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding),
                             nn.BatchNorm2d(channels[1]))
    
    def _stride_pad(self, x, stride):
        batch, channel, freq, time = x.shape
        stride_f = self.stride[0]
        stride_t = self.stride[1]
        pad_f = int(np.ceil((freq-1)/stride_f)*stride_f) + 1 - freq
        pad_t = int(np.ceil((time-1)/stride_t)*stride_t) + 1 - time
        x = torch.cat((x, torch.zeros((batch, channel, freq, pad_t), dtype=x.dtype, device=x.device)),axis=3)
        x = torch.cat((x, torch.zeros((batch, channel, pad_f, time+pad_t), dtype=x.dtype, device=x.device)), axis=2)
        return x
    
    def _kernel_pad(self, kernel_size):
        return [(i - 1) // 2 for i in kernel_size]
    
    def forward(self,xin):
        batch, freq, time = xin.shape
       
        xin = xin.unsqueeze(1)
     
        encoder_out = self.encoder(self._stride_pad(xin, self.stride))
        mix_encoder_out = self.mix_encoder(self._stride_pad(encoder_out, self.mix_stride))
        compressor_out = self.compressor(mix_encoder_out)
        compressor_out = compressor_out.squeeze(1)#(batch, T, F)
        compressor_out = compressor_out.permute(0,2,1)
        blstm_out, _ = self.blstm_block(compressor_out)
        last = self.last_linear(blstm_out)
        mask = last.permute(0,2,1)
        mask = torch.sigmoid(mask)
        return mask

    
if __name__ == '__main__':
    dnn_cfg = {'dnn_cfg': {'f_size': int(2048/2) + 1, 
                      'kernel': (9,9), 
                      'stride': (2,1), 
                      'channel': (1,30),
                      'mix_kernel': (5,15),
                      'mix_stride': (1,1),
                      'mix_channel': (30, 60),
                      'hidden_size': 400}}
    model = FeatExtractorBlstm(dnn_cfg['dnn_cfg'])
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print('parameters:', params)
