import numpy as np
import sys
import glob
import librosa
sys.path.append('../')
import warnings
import os
warnings.filterwarnings('ignore')

class DatasetMaker():
    def __init__(self):
        self.fs = 44100
        self.sec = 6
        self.target_fs = 16000
        self.DSD_folder_path = '../data/DSD100/Sources/'
        self.train_subfolders = glob.glob(self.DSD_folder_path + 'Dev/*')
        self.test_subfolders = glob.glob(self.DSD_folder_path + 'Test/*')
        self.save_dev_folder_path ='../data/DSD_Dev/'
        self.sources_name = ['bass', 'drums', 'other', 'vocals']
        
    def _read_as_mono(self, filename):
        data, _ = librosa.load(filename,  sr=self.fs, dtype=np.float32, mono=True)
        return data
       
    def _downsampilng(self, data):
        y = librosa.resample(data, self.fs, self.target_fs)
        return y
       
    def _is_silent(self, x):
        norm = np.linalg.norm(x, ord=1)
        return  True if norm == 0 else False
    
    def _pad_for_seg_reshape(self, x):
        sig_len = x.shape[0]
        batch_size = np.ceil(sig_len / (self.sec*self.target_fs))
        pad_x = np.zeros(int(batch_size)*(self.sec*self.target_fs))
        pad_x[:sig_len] = x[:]
        return pad_x
    
    def make_dataset(self,type):
        if type == 'Dev':
            subfolders = self.train_subfolders
            save_folder = self.save_dev_folder_path
        
        for n, track_path in enumerate(subfolders):
            print('track_path:', track_path)
    
            for source_name in self.sources_name:
                   source_path = track_path + '/{0}.wav'.format(source_name)
                   data = self._read_as_mono(source_path)
                   ds_data = self._downsampilng(data)
                   ds_data = self._pad_for_seg_reshape(ds_data)
                   ds_seg_data = ds_data.reshape((-1, self.target_fs*self.sec))
                   
                   counter = 0
                   for seg_data in ds_seg_data:
                       if not self._is_silent(seg_data):
                           print('counter:', counter)
                           librosa.output.write_wav(save_folder + source_name + '/{0}_{1}.wav'.format(n, counter), seg_data, self.target_fs)
                           counter += 1
                       else:
                           print("silent data")
                            
if __name__ == '__main__':
    obj = DatasetMaker()
    obj.make_dataset(type='Dev')