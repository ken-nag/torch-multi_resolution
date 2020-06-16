import numpy as np
import torch
import torch.nn as nn

class DemandUNet_p1(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakiness = 0.2        
       
        self.ex1_stride = (1,2)
        self.encoder_stride = [(2,1), (2,2), (2,2), (2,2), (2,2)]
        self.decoder_stride = [(2,2), (2,2), (2,2), (2,2), (2,1)]
        self.last_stride = (1,1)
      
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        self.ex1_kernel = (5,5)
        self.encoder_kernels = [(5,5), (5,5), (5,5), (5,5), (5,5)]
        self.decoder_kernels = [(5,5), (5,5), (5,5), (5,5), (5,5)]
        self.last_kernel = (1,1)
        
        self.ex1_channels = (1,16)
        self.encoder_channels = [(1,16), (32,64), (64,64), (64,64), (64,64)]
        self.decoder_channels = [(64,64), (128,64), (128,64), (128,32), (48,16)]
        self.last_channels = (self.decoder_channels[-1][-1], 1)
        
        self.encoder_depth = len(self.encoder_channels)
        
        self.ex1_encoder = self._encoder_block(in_channels=self.ex1_channels[0],
                                               out_channels=self.ex1_channels[1], 
                                               kernel_size=self.ex1_kernel, 
                                               stride=self.ex1_stride, 
                                               padding=None)
        
        for idx, channel in enumerate(self.encoder_channels):
            enc = self._encoder_block(in_channels=channel[0],
                                      out_channels=channel[1], 
                                      kernel_size=self.encoder_kernels[idx],
                                      stride=self.encoder_stride[idx],
                                      padding=None)
            
            self.add_module("encoder{}".format(idx), enc)
            self.encoders.append(enc)
            
        for idx, channel in enumerate(self.decoder_channels):
            dec = self._decoder_block(in_channels=channel[0],
                                      out_channels=channel[1],
                                      kernel_size=self.decoder_kernels[idx],
                                      stride=self.decoder_stride[idx],
                                      padding=None)
            self.add_module("decoder{}".format(idx), dec)
            self.decoders.append(dec)
            
        self.last_conv=nn.Conv2d(self.last_channels[0], self.last_channels[1], kernel_size=self.last_kernel) 
        
    def _stride_pad(self, x, stride):
        bn, fn, tn = x.shape
        base_fn = 1
        base_tn = 1
        if type(stride) == list:
            for e in stride:
                base_fn = base_fn * e[0]
                base_tn = base_tn * e[1]
        
        if type(stride) == tuple:
            base_fn = base_fn * stride[0]
            base_tn = base_tn * stride[1]
            
        pad_fn = int(np.ceil((fn-1)/base_fn)*base_fn) + 1  - fn
        pad_tn = int(np.ceil((tn-1)/base_tn)*base_tn) + 1  - tn
        x = torch.cat((x, torch.zeros((bn, fn, pad_tn), dtype=x.dtype, device=x.device)),axis=2)
        x = torch.cat((x, torch.zeros((bn, pad_fn, tn+pad_tn), dtype=x.dtype, device=x.device)), axis=1)
        return x, fn, tn
     
    def _kernel_padding(self, kernel_size):
        return [(i - 1) // 2 for i in kernel_size]
    
    def _ex_pad(self, ex_x, x):
        batch, channel, f, t = x.shape
        _, _, ex_f, ex_t = ex_x.shape
        buff = torch.zeros((batch, channel, f, t), dtype=ex_x.dtype, device=ex_x.device)
        buff[:,:,:ex_f, :ex_t] = ex_x[:,:,:,:]
        return buff
    
    def _encoder_block(self,in_channels, out_channels, kernel_size, stride, padding=None, leakiness=0.2):
        if padding is None:
            padding = self._kernel_padding(kernel_size)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(leakiness))
   
    def _decoder_block(self,in_channels,out_channels, kernel_size, stride, padding=None, leakiness=0.2):
        if padding is None:
                padding = self._kernel_padding(kernel_size) # 'same' padding
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU())
                       
    def forward(self, xin, ex1_xin):      
        outputs = []
        ex_outputs = []
        xpad, freqs, frames = self._stride_pad(xin, stride=self.encoder_stride)
        ex1_xpad, ex1_freqs, ex1_frames = self._stride_pad(ex1_xin, stride=self.ex1_stride)
        x = xpad.unsqueeze(1)
        ex1_x = ex1_xpad.unsqueeze(1)
        
        outputs.append(x)
        ex_outputs.append(self.ex1_encoder(ex1_x))
        
        # encoder
        for i in range(len(self.encoders)):
            if i == 1:
                ex_out = self._ex_pad(ex_outputs[i-1], outputs[-1])
                prev_output = torch.cat((outputs[-1], ex_out), dim=1)
            else:
                prev_output = outputs[-1]
            
            outputs.append(self.encoders[i](prev_output))

        # decoder
        prev_output = self.decoders[0](outputs[-1])
        for i in range(1,len(self.decoders)):
            prev_output = self.decoders[i](torch.cat((prev_output, outputs[-i-1]), dim=1))
        est_mask = torch.sigmoid(self.last_conv(prev_output))
        return est_mask[:, 0, :freqs, :frames]

if __name__ == '__main__':
    model = DemandUNet_p1()
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print('parameters:', params)
