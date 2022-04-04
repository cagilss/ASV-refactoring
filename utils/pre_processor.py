import os
import numpy as np 
import librosa  
from random import shuffle
from glob import glob
from sklearn import preprocessing
# import speechpy
from scipy import signal



class PreProcessor:
    def __init__(self):
        self.feature_data = None 
        self.mfcc_path = None
        self.typ = None
    
    def set_typ(self, typ):
        self.typ = typ

    """
    def imp_MFCC(self, files, n_mfcc=20, dct_type=2, 
    n_fft=512, hop_length=128, plot=False): 
        # Mel-frequency cepstral coefficients (MFCCs) 
        
        full_path = os.path.join('PreProcess', 
        'FeatureExtraction', 'MFCC_{}.npy'.format(self.typ))
        
        self.mfcc_path = full_path

        if self.is_exist(full_path):
            return self.load(full_path)

        print('Implementing MFCC to dataset %s... \n' % self.typ)
        data = []
        for file in files:
            signal, sample_rate = librosa.load(file, mono=True)
            mfcc_features = librosa.feature.mfcc(
                y=signal, sr=sample_rate, n_mfcc=n_mfcc, dct_type=dct_type, 
                n_fft=n_fft, hop_length=hop_length)
            swap_features = np.swapaxes(mfcc_features, 0, 1)
            # [nFeatures x time_steps] -> [time_steps x nFeatures] 
            if plot: self.plot_mfcc(mfcc_features)
            data.append(swap_features)
        data_np = np.array(data)

        self.save_dataset(data=data_np, full_path=full_path)
        return data_np
       """

    def get_phraise_bbased(self):
        return 'phraised base wav '

    def concatenate_datasets(self, data_1, data_2, axis=0):
        return np.concatenate((data_1, data_2), axis=axis)

    def load(self, full_path):
        return np.load(full_path, allow_pickle=True)

    def is_exist(self, full_path):
        return os.path.exists(full_path)

    def save_dataset(self, data, full_path):
        np.save(full_path ,data)

    def plot_mfcc(self, mfcc):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()

    def label_data(self, label_file):
        if isinstance(label_file, list):
            y = [self.get_text_file(file) for file in label_file]
            y = np.concatenate(y)
        else:
            y = self.get_text_file(label_file)
            y = np.array(y)
        return y
    
    def get_text_file(self, label_file):
        with open(label_file, 'r') as f:
            f_ = lambda line: 1 if line == 'genuine' else 0
            y = [f_(l.split(' ')[1]) for l in f]
        return np.array(y)
    
    
    # label spoof 
    # div into spoof group in the format of,s {P01: sample_matrix, ... } 
    # return dict 
    def dict_of_spoof_rep_dev(self):
        return 0 

    def separate_tr_ts(self, x, y, freq): 
        idx = int(np.round(x.shape[0] * freq))
        ts_x = x[:idx]              
        ts_y = y[:idx]  
        tr_x = x[idx:]                                                                                                                                                                                          
        tr_y = y[idx:]
        return tr_x, tr_y, ts_x, ts_y

    def zero_mean_norm(self, x): 
        """
        full_path = os.path.join('PreProcess', 
        'ScalingNormalizing', 'zmv_{}.npy'.format(self.typ))
        
        if self.is_exist(full_path):
            return self.load(full_path)
        """
        """
        if x.ndim is 1:
            x, y, f_id = self.transform_into_single_matrix(x, y)
            one_mtx = True
        """
        x_mean = np.mean(x, axis=0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        x_var = np.var(x, axis=0)
        #x_std = np.std(x, axis=0)
        norm_data = (x - x_mean) / x_var
        """
        if one_mtx:
            x, y = self.divide_sample_based(norm_data, y, f_id)
        self.save_dataset(x, full_path)
        """
        return norm_data

    def zero_mean_unit_variance(self, data_x):
        x_scale = preprocessing.scale(data_x)
        return x_scale

    def min_max_scaler(self, x):
        min_max_scaler = preprocessing.MinMaxScaler()
        x_minmax = min_max_scaler.fit_transform(x)
        return x_minmax

    def cepstral_mean_variance_norm(self, vec, variance_normalization=True):
        """ This function is aimed to perform global cepstral mean and
            variance normalization (CMVN) on input feature vector "vec".
            The code assumes that there is one observation per row.

        Args:
            vec (array): input feature matrix
                (size:(num_observation,num_features))
            variance_normalization (bool): If the variance
                normilization should be performed or not.

        Return:
            array: The mean(or mean+variance) normalized feature vector.
        """
        """
        full_path = os.path.join('PreProcess', 
        'ScalingNormalizing', 'cmvn_{}.npy'.format(self.typ))
        
        if self.is_exist(full_path):
            return self.load(full_path)
        """
   
        eps = 2**-30
        rows, cols = vec.shape
        
        # Mean calculation
        norm = np.mean(vec, axis=0)
        norm_vec = np.tile(norm, (rows, 1))

        # Mean subtraction
        mean_subtracted = vec - norm_vec

        # Variance normalization
        if variance_normalization:
            stdev = np.std(mean_subtracted, axis=0)
            stdev_vec = np.tile(stdev, (rows, 1))
            output = mean_subtracted / (stdev_vec + eps)
        else:
            output = mean_subtracted
            
        return output
    
    def shuffle_dataset(self, x, y): 
        arr = np.arange(len(x))
        if isinstance(x, list):
            x = np.array(x)
        shuffle(arr)
        shf_x = np.array(x[arr])
        shf_y = np.array(y[arr])
        return shf_x, shf_y

    def split_batch_size(self, x, y, batch_size=32):
        s = int(np.round(len(x) / batch_size))
        batch_x = np.array(np.array_split(x, s))
        batch_y = np.array(np.array_split(y, s))
        return batch_x, batch_y

    def divide_sample_based(self, x_mtx, y_mtx, id): # optimize method
        x, y = [], []
        f = lambda cond: 1 if cond else 0
        for s in id.values():
            x.append(x_mtx[0:s])
            y.append(f(max(y_mtx[0:s]) == 1))
            x_mtx = x_mtx[s:]
            y_mtx = y_mtx[s:]
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y

    # if time varies in each sample
    def transform_into_single_matrix(self, x, y):
        # id_dic = {0: 122, 1: 345, 2: 98, ... 3013: 456}
        # sum all times and multiply with 3014, however, it might not work 
        x_mtx = np.array([]).reshape(-1, x[0].shape[1])
        y_mtx = np.array([]).reshape(-1, 1)
        frame_id = {}
        count = 0
        for d, l in zip(x, y):
            rep_s = d.shape[0]
            rep_n = np.repeat(l, rep_s).reshape(-1, 1)
            x_mtx = np.vstack((x_mtx, d))
            y_mtx = np.vstack((y_mtx, rep_n))
            frame_id[count] = rep_s
            count += 1
        return x_mtx, y_mtx, frame_id
    
    def single_matrix_form(self, x):
        if isinstance(x, list):
            x = np.array(x)
        size = x.shape
        samp = size[0]
        time = size[1]
        feat = size[2]
        return x.reshape(samp*time, feat)

    def divided_matrix_form(self, x, time=126):
        times = int(np.round(x.shape[0] / time))
        return np.array_split(x, times)

    def adjust_class_proportion(self, x, y, prop):
        # specify current proportion of the classes
        # enter desired c1 and c2 proportions
        if prop >= 0.10:
            c1, y1, c2, y2 = self.seperate_classes(x, y)
            if len(c1) > len(c2):
                c1, y1 = self.update_bigger_class(
                                    c_b=c1,  
                                    y_b=y1,
                                    c_s=c2.shape[0],  
                                    prop=prop
                                    )
            else:
                c2, y2 = self.update_bigger_class(
                                    c_b=c2,  
                                    y_b=y2,
                                    c_s=c1.shape[0],  
                                    prop=prop
                                    )
            x_con = np.concatenate((c1, c2), axis=0)
            y_con = np.concatenate((y1, y2), axis=0)
            x, y = self.shuffle_dataset(x=x_con, y=y_con)
            return x, y
        else:
            print('Enter prop bigger or equal to 0.10')

    def seperate_classes(self, x, y):
        idx1 = np.where(y == 1)[0]
        idx2 = np.where(y == 0)[0]
        return x[idx1], y[idx1], x[idx2], y[idx2]    

    def update_bigger_class(self, c_b, y_b, c_s, prop):
        num = self.arrange_proportion(c_s, c_b.shape[0], prop)  
        c_b = c_b[0:(c_b.shape[0] + num)]
        y_b = y_b[0:(c_b.shape[0] + num)]
        return c_b, y_b

    def arrange_proportion(self, c_s, c_b, prop, ts=13306):  # ts : total size
        return self.find_prop(c_s, (c_s + c_b), prop, ts)

    def find_prop(self, c_s, size, prop, ts):
        for i in range(ts):
            if round((c_s / (size - i)), 2) == prop:
                return -1 * i

    """ second version of pre processing methods """

    def SNR_filter(self, mfcc, limit=0.1, mode='Train'):
        folder = 'SNR_Values'
        file = 'SNR_Values_{}.txt'.format(mode)
        path = os.path.join(folder, file) 
        d = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                smp = line[0]
                db = int(line[3])
                if db > limit:
                    d[smp] = db
                    

        return 'eliminate low db samples'


    def imp_MFCC_2(self, x, limit_feq=False):
        mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(delta)
        feats = np.concatenate((mfcc, delta, delta2), axis=0)
        return feats
    
    def imp_MFCC_1(self, x):
        return librosa.feature.mfcc(x, sr=16000, n_mfcc=20)

    def extract_imfcc(self, audio):
        sample_rate = 16000
        n_fft = int(25 * sample_rate / 1000)
        hop_length = int(10 * sample_rate / 1000)
        n_imfcc = 13
        f_max = sample_rate / 2

        S = np.abs(librosa.core.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2.0
        mel_basis = librosa.filters.mel(sample_rate, n_fft)
        mel_basis = np.linalg.pinv(mel_basis).T
        mel = np.dot(mel_basis, S)
        S = librosa.power_to_db(mel)
        imfcc = np.dot(self.librosa_dct(n_imfcc, S.shape[0]), S)
        imfcc_delta = librosa.feature.delta(imfcc)
        imfcc_delta_delta = librosa.feature.delta(imfcc_delta)
        feature = np.concatenate((imfcc, imfcc_delta, imfcc_delta_delta), axis=0)
        return feature

    def librosa_dct(self, n_filters, n_input):
        """Discrete cosine transform (DCT type-III) basis.

        .. [1] http://en.wikipedia.org/wiki/Discrete_cosine_transform

        Parameters
        ----------
        n_filters : int > 0 [scalar]
            number of output components (DCT filters)

        n_input : int > 0 [scalar]
            number of input components (frequency bins)

        Returns
        -------
        dct_basis: np.ndarray [shape=(n_filters, n_input)]
            DCT (type-III) basis vectors [1]_

        Examples
        --------
        >>> n_fft = 2048
        >>> dct_filters = librosa.filters.dct(13, 1 + n_fft // 2)
        >>> dct_filters
        array([[ 0.031,  0.031, ...,  0.031,  0.031],
            [ 0.044,  0.044, ..., -0.044, -0.044],
            ...,
            [ 0.044,  0.044, ..., -0.044, -0.044],
            [ 0.044,  0.044, ...,  0.044,  0.044]])

        >>> import matplotlib.pyplot as plt
        >>> plt.figure()
        >>> librosa.display.specshow(dct_filters, x_axis='linear')
        >>> plt.ylabel('DCT function')
        >>> plt.title('DCT filter bank')
        >>> plt.colorbar()
        >>> plt.tight_layout()
        """

        basis = np.empty((n_filters, n_input))
        basis[0, :] = 1.0 / np.sqrt(n_input)

        samples = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

        for i in range(1, n_filters):
            basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

        return basis


    def padding(self, x, max_len=64000):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = (max_len / x_len)+1
        x_repeat = np.repeat(x, num_repeats)
        padded_x = x_repeat[:max_len]   
        return padded_x

    # same with librosa
    def read_file(self, meta):
        import soundfile as sf
        data_x, _ = sf.read(meta)
        #data_y = meta.key
        #return data_x, float(data_y), meta.sys_id
        return data_x

    def fft(self, y, sample_rate=16000):
        p_preemphasis = 0.97
        min_level_db = -100
        num_freq = 1025
        ref_level_db = 20
        frame_length_ms = 20
        frame_shift_ms = 10

        def _normalize(S):
            return np.clip((S - min_level_db) / -min_level_db, 0, 1)

        def preemphasis(x):
            return signal.lfilter([1, -p_preemphasis], [1], x)

        def _stft(y):
            n_fft, hop_length, win_length = _stft_parameters()
            return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)

        def _stft_parameters():
            # n_fft = (num_freq - 1) * 2
            n_fft = 1800
            hop_length = 150
            # hop_length = int(frame_shift_ms / 1000 * sample_rate)
            # win_length = int(frame_length_ms / 1000 * sample_rate)
            win_length = 1500
            return n_fft, hop_length, win_length

        def _amp_to_db(x):
            return 20 * np.log10(np.maximum(1e-5, x))
        #y = librosa.core.load(wav_path, sr=sample_rate)[0]
        D = _stft(preemphasis(y))
        S = _amp_to_db(np.abs(D)) - ref_level_db
        return _normalize(S)

    def log_spectrum(self, x):
        s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
        a = np.abs(s)**2
        #melspect = librosa.feature.melspectrogram(S=a)
        feat = librosa.power_to_db(a)
        return feat