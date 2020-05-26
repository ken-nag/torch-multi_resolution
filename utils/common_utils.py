import torch

class CommonUtils():
    def init_bias2zero(self, model):
        for name, params in model.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(params)
        