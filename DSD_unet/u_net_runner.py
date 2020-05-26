import torch
import sys
import time
sys.path.append('../')
from models.u_net import UNet
from data_utils.dsd100_dataset import DSD100Dataset
from utils.loss import MSE
from utils.visualizer import show_TF_domein_result
import numpy as np
from utils.stft_module import STFTModule
import torchaudio.functional as taF

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class UNetRunner():
    def __init__(self, cfg):
        
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.dtype= torch.float32
        self.eps = 1e-4
        
        self.stft_module = STFTModule(cfg['stft_params'], self.device)
        self.train_data_num = cfg['train_data_num']
        self.valid_data_num = cfg['valid_data_num']
        self.sample_len = cfg['sample_len']
        self.epoch_num = cfg['epoch_num']
        self.train_batch_size = cfg['train_batch_size']
        self.valid_batch_size = cfg['valid_batch_size']
        
        self.train_dataset = DSD100Dataset(data_num=self.train_data_num, sample_len=self.sample_len, folder_type='train')
        self.valid_dataset = DSD100Dataset(data_num=self.valid_data_num, sample_len=self.sample_len, folder_type='validation')
        
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.valid_data_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.valid_batch_size, shuffle=True)
        self.model = UNet().to(self.device)
        self.criterion = MSE()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.save_path = 'results/model/train_dsd_config_1/'
        
    def _preprocess(self, mixture, true):
        mix_spec = self.stft_module.stft(mixture, pad=True)
        mix_phase = mix_spec[:,:,1]
        mix_amp_spec = taF.complex_norm(mix_spec)
        mix_amp_spec = mix_amp_spec[:,1:,:]
        mix_mag_spec = torch.log10(mix_amp_spec + self.eps)
        mix_mag_spec = mix_mag_spec[:,1:,:]
        
        true_spec = self.stft_module.stft(true, pad=True)
        true_amp_spec = taF.complex_norm(true_spec)
        true_amp_spec = true_amp_spec[:,1:,:]
        
        return mix_mag_spec, true_amp_spec, mix_phase, mix_amp_spec
        
    def _postporcess(self, est_sources):
        pass
        
    def _run(self, model, criterion, data_loader, batch_size, mode=None):
        running_loss = 0
        for i, (mixture, _, _, _, vocals) in enumerate(data_loader):
            mixture = mixture.to(self.dtype).to(self.device)
            true = vocals.to(self.dtype).to(self.device)
            mix_mag_spec, true_amp_spec, _, mix_amp_spec = self._preprocess(mixture, true)
            
            model.zero_grad()
            est_mask = model(mix_mag_spec.unsqueeze(1))
            est_source = mix_amp_spec.unsqueeze(1) * est_mask
            
            if mode == 'train' or mode == 'validation':
                loss = 10 * criterion(est_source, true_amp_spec)
                running_loss += loss.data
                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
            
        return (running_loss / (i+1)), est_source, est_mask, mix_amp_spec, true_amp_spec
    
    def train(self):
        train_loss = np.array([])
        valid_loss = np.array([])
        print("start train")
        for epoch in range(self.epoch_num):
            # train
            print('epoch{0}'.format(epoch))
            start = time.time()
            self.model.train()
            tmp_train_loss, est_source, est_mask, mix_amp_spec, true_amp_spec = self._run(self.model, self.criterion, self.train_data_loader, self.train_batch_size, mode='train')
            train_loss = np.append(train_loss, tmp_train_loss.cpu().clone().numpy())
            # validation
            # self.model.eval()
            # with torch.no_grad():
            #    tmp_valid_loss, est_source, est_mask, mix_amp_spec = self._run(self.model, self.criterion, self.valid_data_loader, self.valid_batch_size, mode='validation')
            #    valid_loss = np.append(valid_loss, tmp_valid_loss.cpu().clone().numpy())
                 
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.save_path + 'u_net{0}.ckpt'.format(epoch + 1))
            
            end = time.time()
            print('----excute time: {0}'.format(end - start))
            show_TF_domein_result(train_loss, mix_amp_spec[0,:,:], est_mask[0,0,:,:], est_source[0,0,:,:], true_amp_spec[0,:,:])
                        
if __name__ == '__main__':
    from configs.train_dsd_unet_config_1 import cfg as train_cfg
    obj = UNetRunner(train_cfg)
    obj.train()