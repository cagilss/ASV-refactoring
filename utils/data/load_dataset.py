import os
import numpy as np
import random
from random import shuffle
from utils.data.load_files import LoadFiles
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class LoadDataset:
    def __init__(self, tr_x, tr_y, dev_x, dev_y, eval_x, eval_y):
        self.lf = LoadFiles()
        self.tr = self.load_paths(tr_x, tr_y)
        self.dev = self.load_paths(dev_x, dev_y)
        self.ev = self.load_paths(eval_x, eval_y)

    def load_paths(self, x, y):
        x_f_tr, y_f_tr = self.lf.load_file(x, y)
        return {'x':x_f_tr, 'y': y_f_tr}

class LoadDatasetAbs:
    def __init__(self, tr_x, tr_y, dev_x, dev_y, eval_x, eval_y):
        self.lf = LoadFiles()
        self.tr = self.load_paths(tr_x, tr_y)
        self.dev = self.load_paths(dev_x, dev_y)
        self.ev = self.load_paths(eval_x, eval_y)

    def load_paths(self, x, y):
        x_f_tr, y_f_tr = self.lf.load_file(x, y)
        return {'x':x_f_tr, 'y': y_f_tr}

class LoadDatasetPirated:
    def __init__(self, x, y):
        self.x_path = x
        self.y_path = y
        self.all_data = []
        self.load_data()
        
  
    def load_data(self):

        with open(self.y_path) as f:
            labels = f.readlines()

        for lab in labels:
            key, out, _, _, _, _, _ = lab.split()
            key = key.split('.')[0]

            if out == 'genuine':
                out = 1
            else:
                out = 0

            x_path_full = os.path.join(self.x_path, key + '.npy')
            inp = np.load(x_path_full)
            seq_len = len(inp)
            self.all_data += [(key, inp, out, seq_len)]

def asvdataset_collate_fn_pad(batch):
    sample_id = [item[0] for item in batch]
    in_data = [torch.as_tensor(item[1], dtype=None, device=None)
                for item in batch]
    in_data = pad_sequence(in_data, batch_first=True, padding_value=0.0)
    out_data = [torch.as_tensor(item[2], dtype=None, device=None)
                for item in batch]
    seq_len = [item[3] for item in batch]

    return [sample_id, in_data, out_data, seq_len]

class ASVDatasetTorch(Dataset):
    """Custom ASV dataset."""

    def __init__(self, x_p, y_p):

        dataset_obj = LoadDatasetPirated(x=x_p, y=y_p)
     
        self.all_data_list = dataset_obj.all_data

        self.max_len = -1
        self._calc_max_len()

    def __len__(self):
        return len(self.all_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.all_data_list[idx]

        return sample

    def _calc_max_len(self):
        for i in range(len(self.all_data_list)):
            curr_len = len(self.all_data_list[i][1])
            if curr_len > self.max_len:
                self.max_len = curr_len

    def get_max_len(self):
        return self.max_len

    def get_seq_lengths_dict(self):
        seq_lengths = {}
        seq_id = 3

        for i in range(len(self.all_data_list)):
            idx = self.all_data_list[i][0]
            seq_len = self.all_data_list[i][seq_id]
            seq_lengths[idx] = seq_len

        return seq_lengths

    def get_seq_lengths_list(self):
        seq_lengths = []
        seq_id = 3

        for i in range(len(self.all_data_list)):
            idx = self.all_data_list[i][0]
            seq_len = self.all_data_list[i][seq_id]
            seq_lengths += [(idx, seq_len)]

        return seq_lengths

    def get_seq_lengths(self):
        seq_lengths = []
        seq_id = 3

        for i in range(len(self.all_data_list)):
            seq_len = self.all_data_list[i][seq_id]
            seq_lengths += [seq_len]

        return seq_lengths     


class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)


if __name__ == '__main__':

    x_p = 'cqcc_npy_norm/train'
    y_p = 'cqcc_npy_norm/train.trn.txt'

    tr_dataset = ASVDatasetTorch(x_p, y_p)

    seq_lengths = tr_dataset.get_seq_lengths()         

    batch_size = 32
    bin_size = 1  
    is_shuffle = True
    n_worker = 1

    sampler = BinnedLengthSampler(seq_lengths, batch_size, batch_size *bin_size)

    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, 
                        sampler=sampler, collate_fn=asvdataset_collate_fn_pad, 
                        num_workers=n_worker, pin_memory=True)

    for i_batch, sample_batched in enumerate(tr_dataloader):
        sample_id, inp_data, out_data, seq_len = sample_batched
        a=1