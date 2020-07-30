import torch
import torchaudio
import glob
import os
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class Analizer():
    def __init__(self, folder_type):
        self.dtype = torch.float32
        self.dataset_root = '../data/DSD100subset/Sources'
        self.folder_type = folder_type
        
        if self.folder_type == 'Dev':
            self.clean_root = self.dataset_root + '/Dev/'
              
        if self.folder_type == 'Test':
            self.clean_root = self.dataset_root + '/Test/'
        
        self.folder_paths = glob.glob(self.clean_root + '*')
        
        self.snr_list = []
        self.bass_dB_list = []
        self.drums_dB_list = []
        self.other_dB_list = []
        self.vocals_dB_list = []

    def _cal_snr(self, accompany, vocals):
        p_accompany = accompany.abs().sum()
        p_vocals = vocals.abs().sum()
        snr = 20*torch.log10(p_vocals/p_accompany)
        return snr
    
    def _cal_dB(self, source):
        power = source.abs().sum()
        return 20*torch.log10(power)
    
    def _show_result(self, result_list, name):
        print('{0} mean:'.format(name), result_list.mean())
        print('{0} max:'.format(name), result_list.max())
        print('{0} min:'.format(name), result_list.min())
        
    
    def run(self,):
        print(self.folder_type)
        
        for folder_path in self.folder_paths:
            bass, _ = torchaudio.load(folder_path + '/bass.wav')
            drums, _ = torchaudio.load(folder_path + '/drums.wav')
            other, _ = torchaudio.load(folder_path + '/other.wav')
            vocals, _ = torchaudio.load(folder_path + '/vocals.wav')
            
            bass = bass.squeeze(0).to(self.dtype)
            drums = drums.squeeze(0).to(self.dtype)
            other = other.squeeze(0).to(self.dtype)
            vocals = vocals.squeeze(0).to(self.dtype)        
            accompany = bass + drums + other
            
            snr = self._cal_snr(accompany, vocals)
            bass_dB = self._cal_dB(bass)
            drums_dB = self._cal_dB(drums)
            other_dB = self._cal_dB(other)
            vocals_dB = self._cal_dB(vocals)
            
            self.snr_list.append(snr)
            self.bass_dB_list.append(bass_dB)
            self.drums_dB_list.append(drums_dB)
            self.other_dB_list.append(other_dB)
            self.vocals_dB_list.append(vocals_dB)
            
        
        snr_list = torch.tensor(self.snr_list, dtype=torch.float32)
        bass_dB_list = torch.tensor(self.bass_dB_list, dtype=torch.float32)
        drums_dB_list = torch.tensor(self.drums_dB_list, dtype=torch.float32)
        other_dB_list = torch.tensor(self.other_dB_list, dtype=torch.float32)
        vocals_dB_list = torch.tensor(self.vocals_dB_list, dtype=torch.float32)
        
        bass_dB_list = bass_dB_list.detach().clone().cpu().numpy()
        drums_dB_list = drums_dB_list.detach().clone().cpu().numpy()
        other_dB_list = other_dB_list.detach().clone().cpu().numpy()
        vocals_dB_list = vocals_dB_list.detach().clone().cpu().numpy()
        fig, axes = plt.subplots(1,1)
        axes.hist(bass_list, bins=100)
        
        self._show_result(snr_list, name='snr')
        self._show_result(bass_dB_list, name='bass_dB')
        self._show_result(drums_dB_list, name='drums_dB')
        self._show_result(other_dB_list, name='other_dB')
        self._show_result(vocals_dB_list, name='vocal_dB')
        
if __name__ == '__main__':
    train_obj = Analizer(folder_type='Dev')
    test_obj = Analizer(folder_type='Test')
    
    train_obj.run()
    test_obj.run()