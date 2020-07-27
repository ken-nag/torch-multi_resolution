import torch
import sys
import time
sys.path.append('../')
from models.dsd_blstm2 import BLSTM2
from data_utils.voice_demand_dataset import DSD100Dataset
from data_utils.data_loader import FastDataLoader
from utils.loss import T_MAE
from utils.visualizer import show_TF_domein_result
import numpy as np
from utils.stft_module import STFTModule
from utils.common_utils import EarlyStopping
import torchaudio.functional as taF

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class DSDBLSTM2Runner():
    def __init__(self, cfg):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype= torch.float32
        self.eps = 1e-8
        
        self.stft_module = STFTModule(cfg['stft_params'], self.device)
        self.train_data_num = cfg['train_data_num']
        self.valid_data_num = cfg['valid_data_num']
        self.sample_len = cfg['sample_len']
        self.epoch_num = cfg['epoch_num']
        self.train_batch_size = cfg['train_batch_size']
        self.valid_batch_size = cfg['valid_batch_size']
        self.train_full_data_num = cfg['train_full_data_num']
        self.valid_full_data_num = cfg['valid_full_data_num']
        
        self.train_dataset = DSD100Dataset(data_num=self.train_data_num, 
                                                         full_data_num=self.train_full_data_num,
                                                         sample_len=self.sample_len, 
                                                         folder_type='train',
                                                         augmentation=True)
        
        self.valid_dataset = DSD100Dataset(data_num=self.valid_data_num, 
                                                    full_data_num=self.valid_full_data_num,
                                                    sample_len=self.sample_len, 
                                                    folder_type='validation',
                                                    augmentation=False)
        
        self.train_data_loader = FastDataLoader(self.train_dataset, 
                                                batch_size=self.train_batch_size, 
                                                shuffle=True)
        
        self.valid_data_loader = FastDataLoader(self.valid_dataset, 
                                                batch_size=self.valid_batch_size, 
                                                shuffle=True)
        
        self.model = BLSTM2(cfg['dnn_cfg']).to(self.device)
        self.criterion = T_MAE()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.save_path = cfg['save_path']
        self.early_stopping = EarlyStopping(patience=10)
        
    def _preprocess(self, noisy, clean):
        with torch.no_grad():
            noisy_spec = self.stft_module.stft(noisy, pad=False)           
            noisy_amp_spec = taF.complex_norm(noisy_spec)
            noisy_mag_spec = self.stft_module.to_normalize_mag(noisy_amp_spec)
            clean_spec = self.stft_module.stft(clean, pad=False) 
            clean_amp_spec = taF.complex_norm(clean_spec)
        
        return noisy_mag_spec, clean_spec, noisy_spec, noisy_amp_spec, clean_amp_spec
        
        
    def _run(self, mode=None, data_loader=None):
        running_loss = 0
        for i, (noisy, clean) in enumerate(data_loader):
            noisy = noisy.to(self.dtype).to(self.device)
            clean = clean.to(self.dtype).to(self.device)
            noisy_mag_spec, clean_spec, noisy_spec, noisy_amp_spec, clean_amp_spec = self._preprocess(noisy, clean)            
            self.model.zero_grad()
            est_mask = self.model(noisy_mag_spec)
            est_source = noisy_spec * est_mask[...,None]
            
            if mode == 'train' or mode == 'validation':
                loss = self.criterion(est_source, clean_spec, self.stft_module)
                running_loss += loss.data
                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
            
        return (running_loss / (i+1)), est_source, est_mask, noisy_amp_spec, clean_amp_spec
    
    def train(self):
        train_loss = np.array([])
        valid_loss = np.array([])
        print("start train")
        for epoch in range(self.epoch_num):
            # train
            print('epoch{0}'.format(epoch))
            start = time.time()
            self.model.train()
            tmp_train_loss, _, _, _, _ = self._run(mode='train', data_loader=self.train_data_loader)
            train_loss = np.append(train_loss, 
                                   tmp_train_loss.cpu().clone().numpy())
            
            #validation
            self.model.eval()
            with torch.no_grad():
                tmp_valid_loss, est_source, est_mask, noisy_amp_spec, clean_amp_spec = self._run(mode='validation', data_loader=self.valid_data_loader)
                valid_loss = np.append(valid_loss, 
                                       tmp_valid_loss.cpu().clone().numpy())
                
            if (epoch + 1) % 10 == 0:
                plot_time = time.time()
                est_source = taF.complex_norm(est_source)
                show_TF_domein_result(train_loss, 
                                      valid_loss, 
                                      noisy_amp_spec[0,:,:],
                                      est_mask[0,:,:],
                                      est_source[0,:,:],
                                      clean_amp_spec[0,:,:])
                print('plot_time:', time.time() - plot_time)
                torch.save(self.model.state_dict(), self.save_path + 'u_net{0}.ckpt'.format(epoch + 1))
            
            end = time.time()
            print('----excute time: {0}'.format(end - start))
                        
if __name__ == '__main__':
    from configs.dsd_blstm2_config_2 import train_cfg
    obj = DSDBLSTM2Runner(train_cfg)
    obj.train()