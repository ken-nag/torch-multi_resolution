import torch

class PSA():
    def __call__(self, est, true):
        return torch.mean(torch.sum(torch.pow(est - true, 2), [-3,-2,-1]))
     
class MSE():
    def _cal_err(self,est_source, true_source):
        return torch.mean((est_source - true_source)**2, dim=(1,2))
    
    def __call__(self, est_source, true_source):
        batch_size, _, _, _ = est_source.shape
        est_source = est_source.squeeze(1)
        mse_val= self._cal_err(est_source, true_source)
        loss = torch.sum(mse_val) / batch_size
        return loss
    
