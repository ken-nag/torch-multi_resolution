import torch
import torchaudio
import numpy as np

class STFTModule():
    def __init__(self, stft_params, device):
        self.device = device
        self.dtype= torch.float32
        self.n_fft = stft_params['n_fft']
        self.hop_length = stft_params['hop_length']
        self.win_length = stft_params['win_length']
        self.window = torch.hann_window(self.win_length).to(self.dtype).to(self.device)
        self.freq_num = self._cal_freq_num()
        self.pad = None
        self.pad_len = None
        self.sample_len = None
        
    def _cal_freq_num(self):
        return (np.floor(self.n_fft / 2) + 1).astype(np.int32)
        
    def stft(self, x, pad=None):
        center = True
        if pad:
            self.pad = True
            x = self._stft_zero_pad(x)
            center = False
            
        return torch.stft(x, 
                          n_fft=self.n_fft,
                          hop_length=self.hop_length, 
                          win_length=self.win_length,
                          center=center, 
                          window=self.window)
    
    def _stft_zero_pad(self, x):
        batch_size, self.sample_len = x.shape
        frame_num = self._cal_frame_num(self.sample_len)
        pad_x_len = self.win_length + ((frame_num - 1) * self.hop_length)
        self.pad_len = pad_x_len - self.sample_len
        buff = torch.zeros((batch_size, pad_x_len), dtype=self.dtype, device=self.device.type)
        buff[:, :self.sample_len] = x
        return buff
       
    def _cal_frame_num(self, sig_len):
        return np.ceil((sig_len - self.win_length + self.hop_length) / self.hop_length).astype(np.int32)
          
    def _istft_zero_pad(self, x):
        batch_size, f_num, frame_num, channel = x.shape
        pad_size = self.win_length // self.hop_length
        half_pad = pad_size // 2
        pad_x = torch.zeros((batch_size, f_num, frame_num+pad_size, channel), dtype=self.dtype, device=self.device.type)
        pad_x[:,:,half_pad:-half_pad,:] = x[:,:,:,:]
        return pad_x
    
    def _squeeze_istft_pad(self, x):
        half_nfft = self.n_fft // 2
        return x[:, half_nfft:-half_nfft]
        
    def istft(self, x):
        if self.pad:
            x = self._istft_zero_pad(x)
            
        wave  = torchaudio.functional.istft(x, 
                                            n_fft=self.n_fft,
                                            win_length=self.win_length, 
                                            hop_length=self.hop_length,
                                            window=self.window,)
        return wave
            
    def to_normalize_mag(self, x):
        flooring = 1e-4
        eps = 1e-8
        max_val = x.reshape(-1).max()
        logx = torch.log10(torch.clamp(x, min=flooring*max_val))
        norm_logx = (logx - logx.mean(-1, keepdim=True))/(logx.std(-1, keepdim=True)+eps)
        return norm_logx
        
    
    