import torch

class PSA():
    def __call__(self, est, true):
        return torch.mean(torch.sum(torch.pow(est - true, 2), [-3,-2,-1]))
     
class MSE():
    def _cal_err(self,est_source, true_source):
        return torch.mean((est_source - true_source)**2, dim=(1,2))
    
    def __call__(self, est_source, true_source):
        batch_size, _, _, _ = est_source.shape
        mse_val= self._cal_err(est_source, true_source)
        loss = torch.sum(mse_val) / batch_size
        return loss
    
class T_MAE():
    def __call__(self, est_spec, true_spec, stft_module):
        est_wave = stft_module.istft(true_spec)
        true_wave = stft_module.istft(est_spec)
        _, sig_len = est_wave.shape
        return torch.sum(torch.abs(est_wave - true_wave))/sig_len
        
    
