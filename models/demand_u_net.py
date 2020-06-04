import numpy as np
import torch
import torch.nn as nn

class DemandUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = (5,5)
        self.stride = (2,2)
        self.leakiness = 0.2
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.encoder_channels = [(1,32), (32,64), (64,64), (64, 64), (64,64)]
        self.decoder_channels = [(64,64), (128,64), (128, 64), (128,32), (64,16)]
        for idx, channel in enumerate(self.encoder_channels):
            enc = self._encoder_block(in_channels=channel[0],
                                      out_channels=channel[1], 
                                      kernel_size=self.kernel_size,
                                      stride=self.stride)
            self.add_module("encoder{}".format(idx), enc)
            self.encoders.append(enc)
            
        for idx, channel in enumerate(self.decoder_channels):
            dec = self._decoder_block(in_channels=channel[0],
                                      out_channels=channel[1],
                                      kernel_size=self.kernel_size,
                                      stride=self.stride)
            self.add_module("decoder{}".format(idx), dec)
            self.decoders.append(dec)
            
        self.last_conv=nn.Conv2d(self.decoder_channels[-1][-1], 1, kernel_size=1) 
        
    def pre_pad(self, x):
        bn, fn, tn = x.shape
        base_fn = (self.stride[0])**len(self.encoders)
        base_tn = (self.stride[1])**len(self.encoders)
        pad_fn = int(np.ceil((fn-1)/base_fn)*base_fn) + 1 - fn
        pad_tn = int(np.ceil((tn-1)/base_tn)*base_tn) + 1 - tn
        x = torch.cat((x, torch.zeros((bn, fn, pad_tn), dtype=x.dtype, device=x.device)),axis=2)
        x = torch.cat((x, torch.zeros((bn, pad_fn, tn+pad_tn), dtype=x.dtype, device=x.device)), axis=1)
        return x, fn, tn
     
    def _padding(self, kernel_size):
        return [(i - 1) // 2 for i in kernel_size]
    
    def _encoder_block(self,in_channels, out_channels, kernel_size, stride, padding=None, leakiness=0.2):
        if padding is None:
            padding = self._padding(kernel_size)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(leakiness))
   
    def _decoder_block(self,in_channels,out_channels, kernel_size, stride, padding=None, leakiness=0.2):
        if padding is None:
                padding = self._padding(kernel_size) # 'same' padding
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU())
                       
    def forward(self, xin):      
        outputs = []
        xpad, freqs, frames = self.pre_pad(xin)
        x = xpad[:, None, :, :]
        # encoder
        for i in range(len(self.encoders)):
            outputs.append(x)
            x = self.encoders[i](outputs[-1])
        
        # decoder
        x = self.decoders[0](x)
        for i in range(1,len(self.decoders)):
            x = self.decoders[i](torch.cat((x, outputs[-i]), dim=1))
        est_mask = torch.sigmoid(self.last_conv(x))
        return est_mask[:, 0, :freqs, :frames]

   
