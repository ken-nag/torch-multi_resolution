import torch
import sys
import time
import mir_eval
sys.path.append('../')
from models.u_net import UNet
from data_utils.dsd100_dataset import DSD100Dataset
from data_utils.data_loader import FastDataLoader
from utils.stft_module import STFTModule
import torchaudio.functional as taF

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class UNetTester():
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype= torch.float32
        self.eps = 1e-4
        self.eval_path = cfg['eval_path']
        
        self.model = UNet().to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(self.eval_path, map_location=self.device))
        
        self.stft_module = STFTModule(cfg['stft_params'], self.device, )
        self.test_data_num = cfg['test_data_num']
        self.test_batch_size = cfg['test_batch_size']
        self.sample_len = cfg['sample_len']
        self.test_dataset = DSD100Dataset(data_num=self.test_data_num, sample_len=self.sample_len, folder_type='test')
        self.test_data_loader =  FastDataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=True)
        
        self.sdr_list = []
        self.sar_list = []
        self.sir_list = []
            
    def _preprocess(self, mixture, true):
        mix_spec = self.stft_module.stft(mixture, pad=True)
        mix_amp_spec = taF.complex_norm(mix_spec)
        mix_amp_spec = mix_amp_spec[:,1:,:]
        mix_mag_spec = torch.log10(mix_amp_spec + self.eps)
        mix_mag_spec = mix_mag_spec[:,1:,:]
        return mix_mag_spec, mix_spec
    
    def _postprocess(self, x):
        x = x.squeeze(1)
        batch_size, f_size, t_size = x.shape
        pad_x = torch.zeros((batch_size, f_size+1, t_size), dtype=self.dtype, device=self.device)
        pad_x[:,1:, :] = x[:,:,:]
        return pad_x
        
    
    def test(self, mode='test'):
        with torch.no_grad():
            for i, (mixture, _, _, _, vocals) in enumerate(self.test_data_loader):
                start = time.time()
                mixture = mixture.squeeze(0).to(self.dtype).to(self.device)
                true = vocals.squeeze(0).to(self.dtype).to(self.device)
                accompany = mixture - true
                mix_mag_spec, mix_spec = self._preprocess(mixture, true)
                
                est_mask = self.model(mix_mag_spec.unsqueeze(1))
                est_mask = self._postprocess(est_mask)
                est_source = mix_spec * est_mask[...,None]
                est_wave = self.stft_module.istft(est_source)
                print('test time:', time.time() - start)
        

if __name__ == '__main__':
    from configs.dsd_unet_config_1 import test_cfg
    obj = UNetTester(test_cfg)
    obj.test()