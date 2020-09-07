import torch
import torch.nn as nn
import numpy as np

class FeatExtractorBlstm_p1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.f_size = cfg['f_size']
        
        self.kernel=cfg['kernel']
        self.stride=cfg['stride']
        self.channel=cfg['channel']
        self.dilation=cfg['dilation']

        self.mix_kernel=cfg['mix_kernel']
        self.mix_stride=cfg['mix_stride']
        self.mix_channel=cfg['mix_channel']
        self.mix_dilation=cfg['mix_dilation']
        
        self.first_linear_out=cfg['first_linear_out']
        self.hidden_size = cfg['hidden_size']
        self.leakiness=0.2
        
        
        self.encoder = nn.Sequential(self._encoder(channels=self.channel, 
                                                   kernel_size=self.kernel, 
                                                   stride=self.stride, 
                                                   dilation=self.dilation))
        
        self.mix_encoder = self._encoder(channels=self.mix_channel, 
                                         kernel_size=self.mix_kernel, 
                                         stride=self.mix_stride,
                                         dilation=self.mix_dilation)
        
        self.compressor = self._encoder(channels=(self.mix_channel[1],1),
                                        kernel_size=(1,1),
                                        stride=(1,1))
        
        first_linear_in = int(np.ceil(self.f_size/self.stride[0]))
        self.first_linear = nn.Sequential(nn.Linear(in_features=first_linear_in, 
                                                   out_features=self.first_linear_out),
                                         nn.LeakyReLU(self.leakiness))
       
        self.blstm_block = nn.LSTM(input_size=self.first_linear_out,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True, 
                                   batch_first=True)
        
        self.last_linear = nn.Linear(in_features=self.hidden_size*2,
                                     out_features=self.f_size)
    
        
    def _encoder(self, channels, kernel_size, stride, dilation=1):
        padding = self._kernel_and_dilation_pad(kernel_size, dilation)
        return nn.Sequential(nn.Conv2d(in_channels=channels[0],
                                       out_channels=channels[1],
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       dilation=dilation,
                                       padding=padding),
                             nn.InstanceNorm2d(channels[1]),
                             nn.LeakyReLU(self.leakiness))
        
        
    def _stride_pad(self, x, stride):
        batch, channel, freq, time = x.shape
        stride_f = self.stride[0]
        stride_t = self.stride[1]
        pad_f = int(np.ceil((freq-1)/stride_f)*stride_f) + 1 - freq
        pad_t = int(np.ceil((time-1)/stride_t)*stride_t) + 1 - time
        x = torch.cat((x, torch.zeros((batch, channel, freq, pad_t), dtype=x.dtype, device=x.device)),axis=3)
        x = torch.cat((x, torch.zeros((batch, channel, pad_f, time+pad_t), dtype=x.dtype, device=x.device)), axis=2)
        return x
    
    def _kernel_and_dilation_pad(self, kernel_size, dilation):
        return [((i + (i-1)*(dilation - 1) - 1) // 2) for i in kernel_size]
    

    def forward(self,xin, ex1_xin):
        batch, freq, time = xin.shape
       
        xin = xin.unsqueeze(1)
        ex1_xin = ex1_xin.unsqueeze(1)

        encoder_in = torch.cat((xin, ex1_xin), axis=1)
        encoder_out = self.encoder(self._stride_pad(encoder_in, self.stride))
        mix_encoder_out = self.mix_encoder(self._stride_pad(encoder_out, self.mix_stride))
        compressor_out = self.compressor(mix_encoder_out)
        compressor_out = compressor_out.squeeze(1)#(batch, F, T)
        compressor_out = compressor_out.permute(0,2,1)
        first_linear_out = self.first_linear(compressor_out)
        blstm_out, _ = self.blstm_block(first_linear_out)
        last = self.last_linear(blstm_out)
        mask = last.permute(0,2,1)
        mask = torch.sigmoid(mask)
        return mask
    
