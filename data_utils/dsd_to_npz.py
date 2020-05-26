import numpy as np
import random
import sys
import glob
import librosa
sys.path.append('../')
import time

# ignore librosa warning
import warnings
warnings.filterwarnings('ignore')
# drum, bass, other, vocal, mixtureで保存
class DSDToNpz():
    def __init__(self, train_npz_num, valid_npz_num):
        self.valid_npz_num = valid_npz_num
        self.train_npz_num = train_npz_num
        self.fs = 44100
        self.sec = 6
        self.target_fs = 16000
        self.target_len = 16384
        self.DSD_folder_path = '../data/DSD100/Sources/'
        self.save_folder_path ='../data/DSD100npz/'
        
        self.train_subfolders = glob.glob(self.DSD_folder_path + 'Dev/*')
        self.test_subfolders = glob.glob(self.DSD_folder_path + 'Test/*')
        self.sources_name = ['bass', 'drums', 'other', 'vocals']
        
    def _read_as_mono(self, filename):
        data, _ = librosa.load(filename,  sr=self.fs, dtype=np.float32, mono=True)
        return data
        
    def _downsampilng(self, data):
        y = librosa.resample(data, self.fs, self.target_fs)
        return y
        
    def _to_npz(self, file_path, mixture, sources):
        np.savez(file_path, mixture=mixture, sources=sources)
        
    def _silent_exist(self, x):
        d = x.shape[1] - self.target_len
        target_x = x[:,d//2:-d+(d//2)]
        norm = np.linalg.norm(target_x, ord=1, axis=1)
        return  True if np.any(norm == 0) else False
    
    def _random_cutting(self, sources):
        print('sources shape:', sources.shape)
        sources_num = sources.shape[0]
        source_len = sources.shape[1]
        cut_sources_array = np.zeros((sources_num, self.fs*self.sec))
        offset = random.randrange(source_len - self.fs*self.sec)
        cut_sources_array[:, :] = sources[:, offset:offset+self.fs*self.sec]
        return cut_sources_array
    
    def _is_unique_shape(self, source_list):
        return len(set(source_list)) == 1
    
    def make_train_tracks(self, mode='train'):
        assert mode == 'train' or mode == 'validation', 'InvalidArgument'
        
        npz_num = self.train_npz_num if mode == 'train' else self.valid_npz_num
        print('train_subfolders',len(self.train_subfolders))
        num_npz_per_track = npz_num / len(self.train_subfolders) + 1
        
        npz_idx = 0
        for track_path in self.train_subfolders:
            print('track_path:', track_path)
           
            sources_list = []
            sources_shape = []
            for i, source_name in enumerate(self.sources_name):
                source_path = track_path + '/{0}.wav'.format(source_name)
                data = self._read_as_mono(source_path)
                sources_shape.append(data.shape)
                sources_list.append(data)
                sources_shape.append(data.shape)
                print(sources_shape)
                
            if not self._is_unique_shape(sources_shape):
                continue
                             
            npz_num = 0
            while npz_num < num_npz_per_track:
                start = time.time()
                cut_sources = self._random_cutting(np.array(sources_list))
                
                ds_cut_sources = np.zeros((4, (self.target_fs*self.sec)))
                for i, cut_source in enumerate(cut_sources):    
                    ds_cut_source = self._downsampilng(cut_source)
                    ds_cut_sources[i, :] = ds_cut_source[:]
                 
                if not self._silent_exist(ds_cut_sources):
                    npz_num = npz_num + 1
                    mixture = np.sum(ds_cut_sources, axis=0)
                    file_path = self.save_folder_path + '/{0}/{1}{2}'.format(mode, mode, npz_idx) 
                    np.savez(file_path, 
                             mixture=mixture, 
                             bass=ds_cut_sources[0,:], 
                             drums=ds_cut_sources[1,:], 
                             other=ds_cut_sources[2,:],
                             vocals=ds_cut_sources[3,:])#['bass', 'drums', 'other', 'vocals']
                    
                    print('\033[32m' + "create npz:", self.save_folder_path + '{0}/{1}{2}'.format(mode, mode, npz_idx) + '\033[0m')
                    npz_idx = npz_idx + 1
                else:
                    print("\033[31m" + "silent file exists!" + "\033[0m")
                    
                end = time.time()
                print("----excute_time:", end - start)
                    
                                  
    def make_test_tracks(self):
        npz_idx = 0
        black_list = []
        for track_path in self.test_subfolders:
            start = time.time()
            sources_list = []
            sources_shape = []
            for source_name in self.sources_name:
                source_path = track_path + '/{0}.wav'.format(source_name)
                data = self._read_as_mono(source_path)
                sources_list.append(self._downsampilng(data))
                sources_shape.append(data.shape)
                
            if not self._is_unique_shape(sources_shape):
                black_list.append(track_path)
                print('skip:{0}'.format(track_path))
                continue
            
            np_sources = np.array(sources_list)
            mixture = np.sum(np_sources, axis=0)
            file_path = self.save_folder_path + 'test/test{0}'.format(npz_idx) 
            np.savez(file_path, 
                             mixture=mixture, 
                             bass=np_sources[0,:], 
                             drums=np_sources[1,:], 
                             other=np_sources[2,:],
                             vocals=np_sources[3,:])#['bass', 'drums', 'other', 'vocals']
            print("save file: {0}".format(file_path))
            npz_idx = npz_idx + 1
            end = time.time()
            print('---excute_itme:', end - start)
        print('black list:', black_list)
            
if __name__ == '__main__':
    random.seed(0)
    obj = DSDToNpz(train_npz_num=20000, valid_npz_num=2000)
    obj.make_train_tracks(mode='train')
    obj.make_train_tracks(mode='validation')
    obj.make_test_tracks()
