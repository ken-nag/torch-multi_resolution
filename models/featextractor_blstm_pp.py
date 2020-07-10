import torch
import torch.nn as nn
import numpy as np

class FeatExtractorBlstm_pp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.f_size = cfg['f_size']
        
        self.kernel = cfg['kernel']
        self.stride = cfg['stride']
        self.channel = cfg['channel']

        self.ex1_kernel=cfg['ex1_kernel']
        self.ex1_stride=cfg['ex1_stride']
        self.ex1_channel=cfg['ex1_channel']
        self.ex1_dilation=cfg['ex1_dilation']
        
        self.ex2_kernel=cfg['ex2_kernel']
        self.ex2_stride=cfg['ex2_stride']
        self.ex2_channel=cfg['ex2_channel']
        self.ex2_dilation=cfg['ex2_dilation']
        
        self.mix_kernel=cfg['mix_kernel']
        self.mix_stride=cfg['mix_stride']
        self.mix_channel=cfg['mix_channel']
        self.mix_dilation=cfg['mix_dilation']
        
        self.first_linear_out=cfg['first_linear_out']
        self.hidden_size = cfg['hidden_size']
        
        self.encoder = self._encoder(channels=self.channel, kernel_size=self.kernel, stride=self.stride, dilation=self.dilation)
        self.ex1_encoder = self._encoder(channels=self.ex1_channel, kernel_size=self.ex1_kernel, stride=self.ex1_stride, dilation=self.ex1_dilation)
        self.ex2_encoder = self._encoder(channels=self.ex1_channel, kernel_size=self.ex2_kernel, stride=self.ex2_stride, dilation=self.ex2_dilation)
        self.mix_encoder = self._encoder(channels=self.mix_channel, kernel_size=self.mix_stride, stride=self.mix_stride, dilation=self.mix_dilation)
        self.compressor = self._encoder(channels=(self.mix_channel[1],1), kernel_size=(1,1), stride=(1,1))
        first_linear_in = int(np.ceil(self.f_size/self.stride[0]))
        self.first_linear = nn.Linear(in_features=first_linear_in, out_features=self.first_linear_out)
        self.blstm_block = nn.LSTM(input_size=self.first_linear_out,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True, 
                                   batch_first=True)
        
        self.last_linear = nn.Linear(in_features=self.hidden_size*2, out_features=self.f_size)
    
        
    def _encoder(self, channels, kernel_size, stride, dilation=1):
        padding = self._kernel_and_dilation_pad(kernel_size, dilation)
        return nn.Sequential(nn.Conv2d(in_channels=channels[0],
                                       out_channels=channels[1],
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       dilation=dilation,
                                       padding=padding),
                             nn.InstanceNorm2d(channels[1]))
        
        
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
    

    def forward(self,xin, ex1_xin, ex2_xin):
        batch, freq, time = xin.shape
       
        xin = xin.unsqueeze(1)
        ex1_xin = ex1_xin.unsqueeze(1)
        ex2_xin = ex2_xin.unsqueeze(1)
        
        encoder_out = self.encoder(self._stride_pad(xin, self.stride))
        ex1_encoder_out = self.ex1_encoder(self._stride_pad(ex1_xin, self.ex1_stride))
        ex2_encoder_out = self.ex2_encoder(self._stride_pad(ex2_xin, self.ex2_stride))
        mix_encoder_in = torch.cat((encoder_out, ex1_encoder_out, ex2_encoder_out), axis=1)
        mix_encoder_out = self.mix_encoder(self._stride_pad(mix_encoder_in, self.mix_stride))
        compressor_out = self.compressor(mix_encoder_out)
        compressor_out = compressor_out.squeeze(1)#(batch, T, F)
        compressor_out = compressor_out.permute(0,2,1)
        first_linear_out = self.first_linear(compressor_out)
        blstm_out, _ = self.blstm_block(first_linear_out)
        last = self.last_linear(blstm_out)
        mask = last.permute(0,2,1)
        mask = torch.sigmoid(mask)
        return mask
    
if __name__ == '__main__':
    dnn_cfg = {'dnn_cfg': {'f_size': int(2048/2) + 1, 
                          'kernel': (9,9), 
                          'stride': (2,1), 
                          'channel': (1,10),
                          'ex1_kernel': (27,3),
                          'ex1_stride': (2,1),
                          'ex1_channel': (1,10),
                          'ex2_kernel': (3,27),
                          'ex2_stride': (2,1),
                          'ex2_channel': (1,10),
                          'mix_kernel': (5,15),
                          'mix_stride': (1,1),
                          'mix_channel': (30, 60),
                          'hidden_size': 400}}

    model = FeatExtractorBlstm_pp(dnn_cfg['dnn_cfg'])
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print('parameters:', params)
