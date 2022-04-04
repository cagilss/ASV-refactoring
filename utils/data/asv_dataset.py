from torch import Tensor
# from torchvision import transforms
import numpy as np
from joblib import Parallel, delayed
import librosa
from utils.pre_processor import PreProcessor 
import sys

class ASVDataset:
    def __init__(self, paths, args):
        self.tr      = PreProcessData(paths.tr, args).transform_dataset(shuffle=True, batch=True)
        if args.db:
            print('train dataset pre_processing completed')
        self.dev     = PreProcessData(paths.dev, args).transform_dataset(shuffle=True, batch=True)
        if args.db:
            print('dev dataset pre_processing completed')
        self.eval    = PreProcessData(paths.ev, args).transform_dataset(shuffle=False, batch=True)
        if args.db:
            print('eval dataset pre_processing completed')
    

class PreProcessData:
    def __init__(self, paths, args):
        self.paths = paths
        self.args = args

    def transform_dataset(self, shuffle=False, batch=False):
        prep = PreProcessor()
        if isinstance(self.args.pd, int) :
            padding = prep.padding

        if self.args.ft is 'MFCC':
            feature_extractor = prep.imp_MFCC_2
        if self.args.ft is 'IMFCC':
            feature_extractor = prep.extract_imfcc
        if self.args.ft is 'log_power_spec':
            feature_extractor = prep.log_spectrum
        if self.args.ft is 'fft':
            feature_extractor = prep.fft
        if self.args.ft is 'CQCC':
            feature_extractor = 'directly load the CQCC dataset'

        if self.args.norm is 'cmvn':
            normalizer = prep.cepstral_mean_variance_norm
        if self.args.norm is 'minmax':
            normalizer = prep.min_max_scaler
        if self.args.norm is 'default':
            normalizer = librosa.util.normalize
        if self.args.norm is 'mvn':
            normalizer = prep.zero_mean_norm
        
        if self.args.form is 'torch':
            form = Tensor
        if self.args.form is 'np':
            form = np.array

        # direct wav file of eval, dev or train
        files_meta_x = self.paths['x'] 
        files_meta_y = self.paths['y']
        
        raw_data = list(map(prep.read_file, files_meta_x))
        
        if self.args.ft is not 'CQCC':
            transforms_ = transforms.Compose([
                lambda x: padding(x, max_len=self.args.pd),
                lambda x: feature_extractor(x),
                lambda x: np.swapaxes(x, axis1=1, axis2=0),
                lambda x: form(x)
            ])

        transforms_norm = transforms.Compose([
            lambda x: prep.single_matrix_form(x),
            lambda x: normalizer(x),
            lambda x: prep.divided_matrix_form(x) # change time after change feature extractor
        ])

        if self.args.db:
            print('pre processing dataset')

        data = Parallel(n_jobs=4, prefer='threads')(delayed(transforms_)(x) for x in raw_data)
        label = prep.label_data(files_meta_y)

        if self.args.normalize:
            # set time into divided_matrix_form in prep
            data = Parallel(n_jobs=3, prefer='threads')(delayed(transforms_norm)(x) for x in [data])
            data = data[0]
            print('dataset normalized')

        if shuffle:
            data, label = prep.shuffle_dataset(data, label)
        if batch:
            batch_size = self.args.bs
            data, label = prep.split_batch_size(data, label, batch_size=batch_size)
        
        return {'x': data, 'y': label}
