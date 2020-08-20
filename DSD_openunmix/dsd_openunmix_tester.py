import torch
import sys
import time
sys.path.append('../')
from models.dsd_oepnunmix import OpenUnmix
from data_utils.dsd100_dataset import DSD100Dataset
from data_utils.data_loader import FastDataLoader
from utils.stft_module import STFTModule
from utils.evaluation import sp_enhance_evals
import torchaudio.functional as taF
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class DSDOpenUnmixTester():
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype= torch.float32
        self.eps = 1e-4
        self.eval_path = cfg['eval_path']
        
        self.model =  OpenUnmix(cfg['dnn_cfg']).to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(self.eval_path, map_location=self.device))
        
        self.stft_module = STFTModule(cfg['stft_params'], self.device)
        
        self.test_data_num = cfg['test_data_num']
        self.test_batch_size = cfg['test_batch_size']
        self.sample_len = cfg['sample_len']

        self.test_dataset = DSD100Dataset(data_num=self.test_data_num, 
                                          sample_len=self.sample_len, 
                                          folder_type='Test', 
                                          shuffle=False,
                                          device=self.device,
                                          augmentation=False)
        
        self.test_data_loader =  FastDataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
        
        self.stoi_list = np.array([])
        self.pesq_list = np.array([])
        self.si_sdr_list = np.array([])
        self.si_sdr_improve_list = np.array([])
    
    def _preprocess(self, noisy):
        with torch.no_grad():
            noisy_spec = self.stft_module.stft(noisy, pad=None)
            noisy_amp_spec = taF.complex_norm(noisy_spec)
            noisy_mag_spec = self.stft_module.to_normalize_mag(noisy_amp_spec)
            
            return noisy_mag_spec, noisy_spec
    
            
    def test(self, mode='test'):
        with torch.no_grad():
            for i, i, (noisy, _, _, _, clean) in enumerate(self.test_data_loader):
                start = time.time()
                noisy = noisy.to(self.dtype).to(self.device)
                clean = clean.to(self.dtype).to(self.device)
                siglen = noisy.shape[1]
                noisy_mag_spec, noisy_spec = self._preprocess(noisy)
                est_mask = self.model(noisy_mag_spec)
                est_source = noisy_spec * est_mask[...,None]
                est_wave = self.stft_module.istft(est_source, siglen)
                print(est_wave.shape)
                est_wave = est_wave.squeeze(0)
                clean = clean.squeeze(0)
                noisy = noisy.squeeze(0)
                                
                pesq_val, stoi_val, si_sdr_val, si_sdr_improve = sp_enhance_evals(est_wave, clean, noisy, fs=16000)
                self.pesq_list = np.append(self.pesq_list, pesq_val)
                self.stoi_list = np.append(self.stoi_list, stoi_val)
                self.si_sdr_list = np.append(self.si_sdr_list, si_sdr_val)
                self.si_sdr_improve_list = np.append(self.si_sdr_improve_list, si_sdr_improve)
                print('test time:', time.time() - start)
                
            print('pesq mean:', np.mean(self.pesq_list))
            print('stoi mean:', np.mean(self.stoi_list))
            print('si-sdr mean:', np.mean(self.si_sdr_list))
            print('sdr improve mean:', np.mean(self.si_sdr_improve_list))
            
            print('pesq median:', np.median(self.pesq_list))
            print('stoi median:', np.median(self.stoi_list))
            print('si-sder median:', np.median(self.si_sdr_list))
        

if __name__ == '__main__':
    from configs.dsd_openunmix_config_1 import test_cfg
    obj = DSDOpenUnmixTester(test_cfg)
    obj.test()