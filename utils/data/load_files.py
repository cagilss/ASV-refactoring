import numpy as np
import os
import librosa as lb
from glob import glob
import sys

class LoadFiles:

    def are_datasets_exists(self):
        paths = self.data_paths()
        if not paths: return False 
        for p in paths:
            if not os.path.exists(p): 
                return False
        return True

    def load_file(self, x_path, y_path):
        path_audio, path_label = self.dataset_paths(x_path, y_path)
        if self.path_check(path_audio, path_label):
            return self.get_files(path_audio, path_label)

    def load_file_abs(self, x_path, y_path):
        if self.path_check(x_path, y_path):
            return self.get_files(x_path, y_path)

    def path_check(self, xp, yp):
        return self.are_paths_exists(xp, yp) 
        
    def are_paths_exists(self, xp, yp):
        for xp_, yp_ in zip(xp, yp):
            self.is_path_exist(xp_)
            self.is_path_exist(yp_)
        return True

    def is_path_exist(self, p):
        if not os.path.exists(p): 
            raise FileNotFoundError(p)

    def dataset_paths(self, x_path, y_path):
        return self.get_all_paths(x_path, typ='x'), \
               self.get_all_paths(y_path, typ='y')
        
    def get_all_paths(self, p, typ):
        if type(p) == str: return self.get_full_path(p, typ) 
        return [self.get_full_path(p_, typ) for p_ in p]

    def get_full_path(self, p, typ):
        return self.path_docm(self.path_type(typ), p)

    def path_docm(self, p_typ, p):
        #return os.path.join('ASVspoof_2017_Dataset', p_typ % p)
        return p

    def path_type(self, typ):
        return 'ASVspoof2017_V2_%s' if typ == 'x' else 'ASVspoof2017_V2_%s.txt'

    def get_files(self, path_audio, path_label):
        if len(path_audio) > 1:
            x_files = [self.get_x_file(p) for p in path_audio]
            x_files = np.concatenate(x_files)
            y_files = path_label # get labels in here
        else:
            x_files = self.get_x_file(path_audio[0]) 
            y_files = path_label[0]
        return x_files, y_files

    def get_x_file(self, p): 
        return np.asarray(lb.util.find_files(p, ext='wav', recurse=False))

    def load(self): 
        files = self.data_paths()
        print('All train and test files are loaded' + os.linesep)
        return [np.load(f, allow_pickle=True) for f in files]  

    def data_paths(self): # return as dictionary 
        #paths = glob(os.path.join('Dataset', '*.npy'))
        #dic_p = {p:p for p in paths} 
        return glob(os.path.join('Dataset', '*.npy'))
        
