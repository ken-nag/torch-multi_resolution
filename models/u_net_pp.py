import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_pp(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = (5,5)
        self.stride = (2,2)
        self.leakiness = 0.2
        self.dropout_rate = 0.5
        self.encoder_channels = [(1,8), (16,16), (32,64), (64, 128), (128,256), (256,512)]
        self.decoder_channels = [(512, 256), (512, 128), (256, 64), (128,32), (48,16)]
        self.depth = 6
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        self.ex1_encoder = self._encoder_bolock(dim_in=1, dim_out=8, stride=(1,2))
        self.ex2_encoder = self._encoder_bolock(dim_in=1, dim_out=16, stride=(8,1))
        
        for i, channel in enumerate(self.encoder_channels):
            if i == 0:
                self.encoders.append(self._encoder_bolock(dim_in=channel[0], dim_out=channel[1], stride=(2,1)))
            else:
                self.encoders.append(self._encoder_bolock(dim_in=channel[0], dim_out=channel[1], stride=self.stride))
            
        for i, channel in enumerate(self.decoder_channels):
            drop_out = True if i < 3 else False
            self.decoders.append(self._decoder_block(dim_in=channel[0], dim_out=channel[1], drop_out=drop_out))
            
        self.last_layer = nn.ConvTranspose2d(24, 1, self.kernel_size, stride=(2,1), padding=2)
            
    def _encoder_bolock(self, dim_in, dim_out, stride):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, self.kernel_size, stride, padding=2),
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
            
    def forward(self, input, ex1_input, ex2_input):
        
        outputs = []
        outputs.append(input)
        
        ex_outputs = []
        ex_outputs.append(self.ex1_encoder(ex1_input))
        ex_outputs.append(self.ex2_encoder(ex2_input))
        # encoder
        for i in range(self.depth):
            prev_output = torch.cat((outputs[-1], ex_outputs[i-1]), dim=1) if (i == 1 or i == 2) else outputs[-1]
            tmp = self.encoders[i](prev_output)
            outputs.append(tmp)
        
        # decoder
        prev_output = self.decoders[0](outputs[-1])
        for i in range(self.depth-2):
            prev_output = self.decoders[i+1](torch.cat((prev_output, outputs[-(i+2)]), dim=1))
            
        last_output = self.last_layer(torch.cat((prev_output, outputs[1]), dim=1))
        est_mask = torch.sigmoid(last_output)
        
        return est_mask
        