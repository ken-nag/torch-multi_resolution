import torch
import torch.nn as nn

class BLSTM2(nn.Module):
    def __init__(self, f_size):
        super().__init__()
        self.hidden_size = 400
        self.f_size = f_size
        self.blstm_block = nn.LSTM(input_size=self.f_size,
                                    hidden_size=self.f_size*2,
                                    num_layers=2,
                                    bidirectional=True, 
                                    batch_first=True)
        
        self.last_linear = nn.Linear(in_features=self.f_size*4, out_features=self.f_size)
        
    def forward(self, xin):
        xin = xin.permute(0, 2, 1)
        blstm_out, _  = self.blstm_block(xin)
        last_out = self.last_linear(blstm_out)
        mask = torch.sigmoid(last_out)
        mask = mask.permute(0, 2, 1)
        
        return mask

if __name__ == '__main__':
    model = BLSTM2(257)
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print('parameters:', params)