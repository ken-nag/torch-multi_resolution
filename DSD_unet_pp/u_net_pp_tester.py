import torch
import sys
import time
sys.path.append('../')
from models.u_net_pp import UNet_pp
from data_utils.dsd100_dataset import DSD100Dataset
from data_utils.data_loader import FastDataLoader
from utils.stft_module import STFTModule
from utils.evaluation import mss_evals
import torchaudio.functional as taF
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class UNet_pp_Tester():
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype= torch.float32
        self.eps = 1e-4
        self.eval_path = cfg['eval_path']
        
        self.model = UNet_pp().to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(self.eval_path, map_location=self.device))
        
        self.stft_module = STFTModule(cfg['stft_params'], self.device)
        self.stft_module_ex1 = STFTModule(cfg['stft_params_ex1'], self.device)
        self.stft_module_ex2 = STFTModule(cfg['stft_params_ex2'], self.device)
        
        self.test_data_num = cfg['test_data_num']
        self.test_batch_size = cfg['test_batch_size']
        self.sample_len = cfg['sample_len']
        self.test_dataset = DSD100Dataset(data_num=self.test_data_num, sample_len=self.sample_len, folder_type='test', device=self.device, shuffle=False)
        self.test_data_loader =  FastDataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
        
        self.sdr_list = np.array([])
        self.sar_list = np.array([])
        self.sir_list = np.array([])
    
    def _preprocess(self, mixture, true):
        with torch.no_grad():
            mix_spec = self.stft_module.stft(mixture, pad=True)
            mix_amp_spec = taF.complex_norm(mix_spec)
            mix_amp_spec = mix_amp_spec[:,1:,:]
            mix_mag_spec = torch.log10(mix_amp_spec + self.eps)
            
            #ex1
            ex1_mix_spec = self.stft_module_ex1.stft(mixture, pad=True)
            ex1_mix_amp_spec = taF.complex_norm(ex1_mix_spec)
            ex1_mix_mag_spec = torch.log10(ex1_mix_amp_spec + self.eps)
            ex1_mix_mag_spec = ex1_mix_mag_spec[:,1:,1:513]
            
            #ex2
            ex2_mix_spec = self.stft_module_ex2.stft(mixture, pad=True)
            ex2_mix_amp_spec = taF.complex_norm(ex2_mix_spec)
            ex2_mix_mag_spec = torch.log10(ex2_mix_amp_spec + self.eps)
            ex2_mix_mag_spec = ex2_mix_mag_spec[:,1:,:]
            batch_size, f_size, t_size = ex2_mix_mag_spec.shape
            pad_ex2_mix_mag_spec = torch.zeros((batch_size, f_size, 128), dtype=self.dtype, device=self.device)
            pad_ex2_mix_mag_spec[:,:1024,:127] = ex2_mix_mag_spec[:,:,:]
            
            return mix_mag_spec, ex1_mix_mag_spec, pad_ex2_mix_mag_spec,  mix_spec
        
    
    def _postprocess(self, x):
        x = x.squeeze(1)
        batch_size, f_size, t_size = x.shape
        pad_x = torch.zeros((batch_size, f_size+2, t_size), dtype=self.dtype, device=self.device)
        pad_x[:,1:-1, :] = x[:,:,:]
        return pad_x
        
    
    def test(self, mode='test'):
        with torch.no_grad():
            for i, (mixture, _, _, _, vocals) in enumerate(self.test_data_loader):
                start = time.time()
                mixture = mixture.squeeze(0).to(self.dtype).to(self.device)
                true = vocals.squeeze(0).to(self.dtype).to(self.device)
                
                mix_mag_spec, ex1_mix_mag_spec, ex2_mix_mag_spec, mix_spec = self._preprocess(mixture, true)
                est_mask = self.model(mix_mag_spec.unsqueeze(1), ex1_mix_mag_spec.unsqueeze(1), ex2_mix_mag_spec.unsqueeze(1))
                est_mask = self._postprocess(est_mask)
                est_source = mix_spec * est_mask[...,None]
                est_wave = self.stft_module.istft(est_source)
                
                est_wave = est_wave.flatten()  
                mixture = mixture.flatten()
                true = true.flatten()
                true_accompany = mixture - true
                est_accompany = mixture - est_wave
                sdr, sir, sar = mss_evals(est_wave, est_accompany, true, true_accompany)
                self.sdr_list = np.append(self.sdr_list, sdr)
                self.sar_list = np.append(self.sar_list, sar)
                self.sir_list = np.append(self.sir_list, sir)
                print('test time:', time.time() - start)
                
            print('sdr mean:', np.mean(self.sdr_list))
            print('sir mean:', np.mean(self.sir_list))
            print('sar mean:', np.mean(self.sar_list))
        

if __name__ == '__main__':
    from configs.dsd_unet_pp_config_1 import test_cfg
    obj = UNet_pp_Tester(test_cfg)
    obj.test()