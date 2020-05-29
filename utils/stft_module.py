import torch
import torchaudio
from scipy.signal import hann
import numpy as np

class STFTModule():
    def __init__(self, stft_params, device):
        self.device = device
        self.dtype= torch.float32
        self.n_fft = stft_params['n_fft']
        self.hop_length = stft_params['hop_length']
        self.win_length = stft_params['win_length']
        self.window = torch.from_numpy(hann(self.win_length)).to(self.dtype).to(self.device)
        self.freq_num = self._cal_freq_num()
        self.pad = None
        self.pad_len = None
        self.sample_len = None
        
    def _cal_freq_num(self):
        return (np.floor(self.n_fft / 2) + 1).astype(np.int32)
        
    def stft(self, x, pad=None):
        if pad:
            self.pad = pad
            x = self._zero_pad(x)
            
        return torch.stft(x, 
                          n_fft=self.n_fft,
                          hop_length=self.hop_length, 
                          win_length=self.win_length,
                          center=None, 
                          window=self.window)
    
    def _zero_pad(self, x):
        batch_size, self.sample_len = x.shape
        frame_num = self._cal_frame_num(self.sample_len)
        pad_x_len = self.win_length + ((frame_num - 1) * self.hop_length)
        self.pad_len = pad_x_len - self.sample_len
        buff = torch.zeros((batch_size, pad_x_len), dtype=self.dtype, device=self.device.type)
        buff[:, :self.sample_len] = x
        return buff
       
    def _cal_frame_num(self, sig_len):
        return np.ceil((sig_len - self.win_length + self.hop_length) / self.hop_length).astype(np.int32)
      
    def _squeeze_pad(self):
        pass
     
    def istft(self, x):
        wave  = torchaudio.functional.istft(x, 
                                            n_fft=self.n_fft,
                                            win_length=self.win_length, 
                                            hop_length=self.hop_length,
                                            window=self.window,
                                            center=None,
                                            length=self.sample_len+self.pad_len)
        return wave
            
    def stft_3D(self, x, pad=None):
       batch_size, source_num, sig_len = x.shape
       frame_num = self._cal_frame_num(sig_len)
       buff = torch.zeros((batch_size, source_num, self.freq_num, frame_num, 2)).to(self.dtype).to(self.device)
       for i, source in enumerate(x):
           buff[i, :, :, :, :] = self.stft(source, pad=pad)
           
       return buff
    
    