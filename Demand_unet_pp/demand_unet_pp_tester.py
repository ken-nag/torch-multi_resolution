import torch
import sys
import time
sys.path.append('../')
from models.u_net_pp import UNet_pp
from data_utils.voice_demand_dataset import VoicebankDemandDataset
from data_utils.data_loader import FastDataLoader
from utils.stft_module import STFTModule
from utils.evaluation import sp_enhance_evals
import torchaudio.functional as taF
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class DemandUNet_pp_Tester():
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
        #tmp
        self.test_dataset = VoicebankDemandDataset(data_num=self.test_data_num, sample_len=self.sample_len, folder_type='train', shuffle=False)
        self.test_data_loader =  FastDataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
        
        self.stoi_list = np.array([])
        self.pesq_list = np.array([])
        self.si_sdr_list = np.array([])
    
    def _preprocess(self, noisy):
        with torch.no_grad():
            noisy_spec = self.stft_module.stft(noisy, pad=True)
            noisy_amp_spec = taF.complex_norm(noisy_spec)
            noisy_amp_spec = noisy_amp_spec[:,1:,:]
            noisy_mag_spec = torch.log10(noisy_amp_spec + self.eps)
            
            #ex1
            ex1_noisy_spec = self.stft_module_ex1.stft(noisy, pad=True)
            ex1_noisy_amp_spec = taF.complex_norm(ex1_noisy_spec)
            ex1_noisy_mag_spec = torch.log10(ex1_noisy_amp_spec + self.eps)
            ex1_noisy_mag_spec = ex1_noisy_mag_spec[:,1:,1:513]
            
            #ex2
            ex2_noisy_spec = self.stft_module_ex2.stft(noisy, pad=True)
            ex2_noisy_amp_spec = taF.complex_norm(ex2_noisy_spec)
            ex2_noisy_mag_spec = torch.log10(ex2_noisy_amp_spec + self.eps)
            ex2_noisy_mag_spec = ex2_noisy_mag_spec[:,1:,:]
            batch_size, f_size, t_size = ex2_noisy_mag_spec.shape
            pad_ex2_noisy_mag_spec = torch.zeros((batch_size, f_size, 128), dtype=self.dtype, device=self.device)
            pad_ex2_noisy_mag_spec[:,:1024,:127] = ex2_noisy_mag_spec[:,:,:]
            
            return noisy_mag_spec, ex1_noisy_mag_spec, pad_ex2_noisy_mag_spec,  noisy_spec
        
    
    def _postprocess(self, x):
        x = x.squeeze(1)
        batch_size, f_size, t_size = x.shape
        pad_x = torch.zeros((batch_size, f_size+2, t_size), dtype=self.dtype, device=self.device)
        pad_x[:,1:-1, :] = x[:,:,:]
        return pad_x
        
    
    def test(self, mode='test'):
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(self.test_data_loader):
                start = time.time()
                noisy = noisy.squeeze(0).to(self.dtype).to(self.device)
                clean = clean.squeeze(0).to(self.dtype).to(self.device)
                noisy_mag_spec, ex1_noisy_mag_spec, ex2_noisy_mag_spec, noisy_spec = self._preprocess(noisy)
                est_mask = self.model(noisy_mag_spec.unsqueeze(1), ex1_noisy_mag_spec.unsqueeze(1), ex2_noisy_mag_spec.unsqueeze(1))
                est_mask = self._postprocess(est_mask)
                est_source = noisy_spec * est_mask[...,None]
                est_wave = self.stft_module.istft(est_source)
                
                est_wave = est_wave.flatten()  
                clean = clean.flatten()
                
                pesq_val, stoi_val, si_sdr_val = sp_enhance_evals(est_wave, clean, fs=16000)
                self.pesq_list = np.append(self.pesq_list, pesq_val)
                self.stoi_list = np.append(self.stoi_list, stoi_val)
                self.si_sdr_list = np.append(self.si_sdr_list, si_sdr_val)
                print('test time:', time.time() - start)
                
            print('pesq mean:', np.mean(self.pesq_list))
            print('stoi mean:', np.mean(self.stoi_list))
            print('sdr mean:', np.mean(self.si_sdr_list))
        

if __name__ == '__main__':
    from configs.demand_unet_pp_config_1 import test_cfg
    
    obj = DemandUNet_pp_Tester(test_cfg)
    obj.test()