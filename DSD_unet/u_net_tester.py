import torch
import sys
import time
import mir_eval
sys.path.append('../')
from models.u_net import UNet
from data_utils.dsd100_dataset import DSD100Dataset
from data_utils.data_loader import FastDataLoader
import numpy as np
from utils.stft_module import STFTModule
import torchaudio.functional as taF

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class UNetTester():
    def __init_(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype= torch.float32
        self.eps = 1e-4
        model = UNet().to(self.device)
        self.model = model.load_state_dict(torch.load(self.eval_path, map_location=self.device))
        self.stft_module = STFTModule(cfg['stft_params'], self.device)
        self.test_data_num = cfg['test_data_num']
        self.sample_len = cfg['sample_len']
        self.test_dataset = DSD100Dataset(data_num=self.train_data_num, sample_len=self.sample_len, folder_type='test')
        self.test_dataloader =  FastDataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=True)
        
    def _preprocess(self, mixture, true):
        with torch.no_grad():
            mix_spec = self.stft_module.stft(mixture, pad=True)
            mix_amp_spec = taF.complex_norm(mix_spec)
            mix_amp_spec = mix_amp_spec[:,1:,:]
            mix_mag_spec = torch.log10(mix_amp_spec + self.eps)
            mix_mag_spec = mix_mag_spec[:,1:,:]
        
        return mix_mag_spec, mix_spec, mix_phase, mix_amp_spec
    
     def _run(self, mode=None, data_loader):
        running_loss = 0
        for i, (mixture, _, _, _, vocals) in enumerate(data_loader):
            print('i_iter;',i)
            print('mixture_shape:', mixture.shape)
            print('vocals_shape:', vocals.shape)
            mixture = mixture.to(self.dtype).to(self.device)
            true = vocals.to(self.dtype).to(self.device)
            mix_mag_spec, ex1_mix_mag_spec, ex2_mix_mag_spec, true_amp_spec, _, mix_amp_spec = self._preprocess(mixture, true)
            
            self.model.zero_grad()
            est_mask = self.model(mix_mag_spec.unsqueeze(1), ex1_mix_mag_spec.unsqueeze(1), ex2_mix_mag_spec.unsqueeze(1))
            est_mask = self._postporcess(est_mask)
            est_source = mix_amp_spec.unsqueeze(1) * est_mask
            
        return (running_loss / (i+1)), est_source, est_mask, mix_amp_spec, true_amp_spec
    
    def test(self):
        self.model.eval()
    

if __name__ == '__main__':
    pass