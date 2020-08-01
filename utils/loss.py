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
        est_wave = stft_module.istft(est_spec)
        true_wave = stft_module.istft(true_spec)
        noise = est_wave - true_wave
        loss = noise.abs().mean(-1)
        return loss.mean()
        
class Clip_SDR():
    def _cal_sdr(self, est_wave, true_wave):
        noise = est_wave - true_wave
        p_noise = noise.pow(2).sum(-1)
        p_true = true_wave.pow(2).sum(-1)
        sdr = 10*torch.log10(p_true/p_noise)
        return sdr
    
    def __call__(self, est_spec, true_spec, stft_module):
        est_wave = stft_module.istft(est_spec)
        true_wave = stft_module.istft(true_spec)
        sdr = self._cal_sdr(est_wave, true_wave)
        a = 20
        clip_sdr = torch.sum(a * torch.tanh(sdr/a))
        return -clip_sdr
