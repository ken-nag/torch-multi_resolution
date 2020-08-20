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
        p_noise = noise.pow(2).sum(-1) + 1e-8
        p_true = true_wave.pow(2).sum(-1) + 1e-8
        sdr = 10*torch.log10(p_true/p_noise)
        return sdr
    
    def __call__(self, est_spec, true_spec, stft_module):
        est_wave = stft_module.istft(est_spec)
        true_wave = stft_module.istft(true_spec)
        sdr = self._cal_sdr(est_wave, true_wave)
        a = 20
        clip_sdr = torch.sum(a * torch.tanh(sdr/a))
        return -clip_sdr

class MultiSTFT_Loss():
    def _time_l1_loss(self, est_wave, true_wave):
        noise = est_wave - true_wave
        return noise.abs().sum(-1)
    
    def _frobenius_norm(self, spec):
        return spec.pow(2).sum([1,2])
    
    def _sc_loss(self, est_spec, true_spec, stft_module):
        noise = est_spec - true_spec
        return self._frobenius_norm(noise)/self._frobenius_norm(true_spec)
    
    def _mag_loss(self, est_spec, true_spec, stft_moduel):
        pass
    
    def _stft_loss(self, est_wave, true_wave, stft_module):
        est_spec = self.stft_module.stft(est_wave, pad=False)   
        true_spec = self.stft_module.stft(true_wave, pad=False)
        sc_loss = self._sc_loss(est_spec, true_spec, stft_module)
        mag_loss = self._mag_loss(est_spec, true_spec, stft_module)
        return sc_loss + mag_loss
        
    
    def __call__(self, est_spec, true_spec, stft_module, stft_module_ex1, stft_module_ex2):
        est_wave = stft_module.istft(est_spec)
        true_wave = stft_module.istft(true_spec)
        batch, sample_len = est_wave.shape
        time_l1_loss = self._time_l1_loss(est_wave, true_wave)
        stft_loss = self._stft_loss(est_wave, true_wave, stft_module)
        ex1_stft_loss = self._stft_loss(est_wave, true_wave, stft_module_ex1)
        ex2_stft_loss = self._stft_loss(est_wave, true_wave, stft_module_ex2)
        return (time_l1_loss + stft_loss + ex1_stft_loss + ex2_stft_loss)/sample_len
        
        