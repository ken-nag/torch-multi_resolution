import torch

class CommonUtils():
    def init_bias2zero(self, model):
        for name, params in model.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(params)
                
                
class EarlyStopping():
    def __init__(self, patience=0):
        self.step = 0
        self.loss = torch.tensor(float('inf'))
        self.patience = patience
    
    def validation(self, loss):
        if self.loss < loss:
            self.step += 1
            if self.step > self.patience:
                print('Early stopping!')
                return True
            
        else:
            self.step = 0
            self.loss = loss
        
        return False