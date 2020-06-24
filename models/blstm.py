import torch
import torch.nn as nn

class BLSTM2(nn.Module):
    def __init__(self, f_size):
        super().__init__()
        self.hidden_size = 400
        self.f_size = f_size
        self.blstm_block = nn.LSTM(input_size=self.f_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=True, 
                                   batch_first=True)
        
        self.last_linear = nn.Linear(in_features=self.hidden_dim*2, out_features=self.f_size)
        
    def forward(self, xin):
        xin = xin.permute(0, 2, 1)
        blstm_out  = self.blstm_block(xin)
        last_out = self.last_linear(blstm_out)
        mask = torch.sigmoid(last_out)
        mask = mask.permute(0, 2, 1)
        
        return mask