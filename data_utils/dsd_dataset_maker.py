import numpy as np
import random
import sys
import glob
import librosa
sys.path.append('../')
import time
import warnings
import os
warnings.filterwarnings('ignore')

class DatasetMaker():
     def __init__(self):
         self.fs = 44100 #要確認
         self.target_fs = 16000
         self.DSD_folder_path = '../data/DSD100/Sources/'
         self.train_subfolders = glob.glob(self.DSD_folder_path + 'Dev/*')
         self.test_subfolders = glob.glob(self.DSD_folder_path + 'Test/*')
         self.save_dev_folder_path ='../data/DSD_Dsampled/Sources/Dev/'
         self.save_test_folder_path = '../data/DSD_Dsampled/Sources/Test/'
         self.sources_name = ['bass', 'drums', 'other', 'vocals']
         
     def _read_as_mono(self, filename):
        data, _ = librosa.load(filename,  sr=self.fs, dtype=np.float32, mono=True)
        return data
    
     def _downsampilng(self, data):
        y = librosa.resample(data, self.fs, self.target_fs)
        return y
    
     def make_dataset(self,type):
         if type == 'Dev':
             subfolders = self.train_subfolders
             save_folder = self.save_dev_folder_path
         if type == 'Test':
             subfolders = self.test_subfolders
             save_folder = self.save_test_folder_path
         
             
         for n, track_path in enumerate(subfolders):
             print('track_path:', track_path)
             save_path = save_path = save_folder + 'track{0}/'.format(n)
             os.mkdir(save_path)
             for i, source_name in enumerate(self.sources_name):
                    source_path = track_path + '/{0}.wav'.format(source_name)
                    data = self._read_as_mono(source_path)
                    ds_data = self._downsampilng(data)
                    librosa.output.write_wav(save_path + source_name + '.wav', ds_data, self.target_fs)
                    
                    

if __name__ == '__main__':
    obj = DatasetMaker()
    obj.make_dataset(type='Dev')