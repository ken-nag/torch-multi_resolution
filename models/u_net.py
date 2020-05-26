import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = (5,5)
        self.stride = (2,2)
        self.leakiness = 0.2
        self.dropout_rate = 0.5
        self.encoder_channels = [(1,16), (16,32), (32,64), (64, 128), (128,256), (256,512)]
        self.decoder_channels = [(512, 256), (512, 128), (256, 64), (128,32), (64,16)]
        self.depth = 6
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for channel in self.encoder_channels:
            self.encoders.append(self._encoder_bolock(dim_in=channel[0], dim_out=channel[1]))
            
        for i, channel in enumerate(self.decoder_channels):
            drop_out = True if i < 3 else False
            self.decoders.append(self._decoder_block(dim_in=channel[0], dim_out=channel[1], drop_out=drop_out))
            
        self.last_layer = nn.ConvTranspose2d(32, 1, self.kernel_size, self.stride, padding=2, output_padding=1)
            
    def _encoder_bolock(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, self.kernel_size, self.stride, padding=2),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(self.leakiness))
    
    def _decoder_block(self, dim_in, dim_out, drop_out):
        if drop_out:
           return nn.Sequential(nn.ConvTranspose2d(dim_in, dim_out, self.kernel_size, self.stride, padding=2, output_padding=1),
                                nn.BatchNorm2d(dim_out),
                                nn.Dropout2d(self.dropout_rate),
                                nn.ReLU())
        else:
           return nn.Sequential(nn.ConvTranspose2d(dim_in, dim_out, self.kernel_size, self.stride, padding=2, output_padding=1),
                                nn.BatchNorm2d(dim_out),
                                nn.ReLU())
            
    def forward(self, input):
        
        outputs = []
        outputs.append(input)
        
        # encoder
        for i in range(self.depth):
            prev_output = outputs[-1]
            tmp = self.encoders[i](prev_output)
            outputs.append(tmp)
        
        # decoder
        prev_output = self.decoders[0](outputs[-1])
        for i in range(self.depth-2):
            prev_output = self.decoders[i+1](torch.cat((prev_output, outputs[-(i+2)]), dim=1))
            
        last_output = self.last_layer(torch.cat((prev_output, outputs[1]), dim=1))
        est_mask = F.sigmoid(last_output)
        
        return est_mask
        