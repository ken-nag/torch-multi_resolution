import torch
import sys
import time
sys.path.append('../')
from models.u_net_pp import UNet_pp
from data_utils.voice_demand_dataset import VoicebankDemandDataset
from data_utils.data_loader import FastDataLoader
from utils.loss import MSE
from utils.visualizer import show_TF_domein_result
import numpy as np
from utils.stft_module import STFTModule
import torchaudio.functional as taF

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


class DemandUNet_pp_Runner():
    def __init__(self, cfg):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype= torch.float32
        self.eps = 1e-4
        
        self.stft_module = STFTModule(cfg['stft_params'], self.device)
        self.stft_module_ex1 = STFTModule(cfg['stft_params_ex1'], self.device)
        self.stft_module_ex2 = STFTModule(cfg['stft_params_ex2'], self.device)
        
        self.train_data_num = cfg['train_data_num']
        self.valid_data_num = cfg['valid_data_num']
        self.sample_len = cfg['sample_len']
        self.epoch_num = cfg['epoch_num']
        self.train_batch_size = cfg['train_batch_size']
        self.valid_batch_size = cfg['valid_batch_size']
        
        self.test_dataset = VoicebankDemandDataset(data_num=self.train_data_num, sample_len=self.sample_len, folder_type='train')
        self.test_data_loader = FastDataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=True)
      
        self.model = UNet_pp().to(self.device)
        self.criterion = MSE()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.save_path = 'results/model/demand_unet_pp_config_1/'
        
    def _preprocess(self, noisy, clean):
        with torch.no_grad():
            noisy_spec = self.stft_module.stft(noisy, pad=True)
            noisy_amp_spec = taF.complex_norm(noisy_spec)
            noisy_amp_spec = noisy_amp_spec[:,1:,:]
            noisy_mag_spec = torch.log10(noisy_amp_spec + self.eps)
            
            clean_spec = self.stft_module.stft(clean, pad=True)
            clean_amp_spec = taF.complex_norm(clean_spec)
            clean_amp_spec = clean_amp_spec[:,1:,:]
            
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
            
            return noisy_mag_spec, ex1_noisy_mag_spec, pad_ex2_noisy_mag_spec, clean_amp_spec, noisy_amp_spec
        
    def _postporcess(self, x):
        #padding DC Component
        batch_size, channel_size, f_size, t_size = x.shape
        pad_x = torch.zeros((batch_size, channel_size, f_size+1, t_size), dtype=self.dtype, device=self.device)
        pad_x[:,:,1:, :] = x[:,:,:,:]
        return pad_x
        
    def _run(self, mode=None, data_loader=None):
        running_loss = 0
        for i, (noisy, clean) in enumerate(data_loader):
            noisy = noisy.to(self.dtype).to(self.device)
            clean = clean.to(self.dtype).to(self.device)
            noisy_mag_spec, ex1_noisy_mag_spec, ex2_noisy_mag_spec, clean_amp_spec,  noisy_amp_spec = self._preprocess(noisy, clean)
            
            self.model.zero_grad()
            est_mask = self.model(noisy_mag_spec.unsqueeze(1), ex1_noisy_mag_spec.unsqueeze(1), ex2_noisy_mag_spec.unsqueeze(1))
            est_mask = self._postporcess(est_mask)
            est_source = noisy_amp_spec.unsqueeze(1) * est_mask
            
            if mode == 'train' or mode == 'validation':
                loss = 10 * self.criterion(est_source, clean_amp_spec)
                running_loss += loss.data
                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
            
        return (running_loss / (i+1)), est_source, est_mask, noisy_amp_spec, clean_amp_spec
    
    def train(self):
        train_loss = np.array([])
        print("start train")
        for epoch in range(self.epoch_num):
            # train
            print('epoch{0}'.format(epoch))
            start = time.time()
            self.model.train()
            tmp_train_loss, est_source, est_mask, noisy_amp_spec, clean_amp_spec = self._run(mode='train', data_loader=self.test_data_loader)
            train_loss = np.append(train_loss, tmp_train_loss.cpu().clone().numpy())
            if (epoch + 1) % 10 == 0:
                show_TF_domein_result(train_loss, noisy_amp_spec[0,:,:], est_mask[0,0,:,:], est_source[0,0,:,:], clean_amp_spec[0,:,:])
                torch.save(self.model.state_dict(), self.save_path + 'u_net{0}.ckpt'.format(epoch + 1))
            
            end = time.time()
            print('----excute time: {0}'.format(end - start))
           
                        
if __name__ == '__main__':
    from configs.demand_unet_pp_config_1 import train_cfg
    obj = DemandUNet_pp_Runner(train_cfg)
    obj.train()