import torch
import torch.nn as nn
import numpy as np

class FeatExtractorBlstm(nn.Module):
    def __init__(self, f_size):
        super().__init__()
        self.f_size = f_size
        self.kernel_size = [(5,5), (5,5)]
        self.stride = [(1,1), (1,1)]
        self.encoder_channels = [(1,30), (30,60)]
        self.encoder_depth = len(self.encoder_channels)
        self.hidden_size = 400
        
        self.encoders = nn.ModuleList()
        for i in range(self.encoder_depth):
            self.encoders.append(self._encoder(self.encoder_channels[i],
                                               self.kernel_size[i],
                                               self.stride[i],))
        
        self.compressor = self._encoder(channels=(60,1), kernel_size=(1,1), stride=(1,1))
        
        self.blstm_block = nn.LSTM(input_size=self.f_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True, 
                                   batch_first=True)
        
        self.last_linear = nn.Linear(in_features=self.hidden_size*2, out_features=self.f_size)
    
        
    def _encoder(self, channels, kernel_size, stride):
        padding = self._padding(kernel_size)
        return nn.Conv2d(in_channels=channels[0],
                         out_channels=channels[1],
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)
        
        
    def _pre_pad(self, x):
        bn, fn, tn = x.shape
        base_fn = (self.stride[0][0])**(len(self.encoders))
        base_tn = (self.stride[0][1])**(len(self.encoders))
        pad_fn = int(np.ceil((fn-1)/base_fn)*base_fn) + 1 - fn
        pad_tn = int(np.ceil((tn-1)/base_tn)*base_tn) + 1 - tn
        x = torch.cat((x, torch.zeros((bn, fn, pad_tn), dtype=x.dtype, device=x.device)),axis=2)
        x = torch.cat((x, torch.zeros((bn, pad_fn, tn+pad_tn), dtype=x.dtype, device=x.device)), axis=1)
        return x, fn, tn
    
    def _padding(self, kernel_size):
        return [(i - 1) // 2 for i in kernel_size]
        
    def forward(self,xin):
        xpad, freqs, frames = self._pre_pad(xin)
        xpad = xpad.permute(0,2,1)#(batch, T, F)
        xpad  = xpad.unsqueeze(1)#(batch, channel T, F)
        encoder_out = xpad
        for i in range(self.encoder_depth):   
            encoder_out = self.encoders[i](encoder_out)
        
        compressor_out = self.compressor(encoder_out)
        compressor_out = compressor_out.squeeze(1)#(batch, T, F)
        blstm_out, _ = self.blstm_block(compressor_out)
        
        last = self.last_linear(blstm_out)
        mask = last.permute(0,2,1)
        mask = torch.sigmoid(mask)
        return mask
    
if __name__ == '__main__':
    model = FeatExtractorBlstm(513)
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print('parameters:', params)
