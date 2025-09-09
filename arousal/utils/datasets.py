import torch
import torch.nn.functional as F
import pickle
import numpy as np
import os

from torch.utils.data import Dataset


def spec_collate_fn(batch_list):
    # 1) freq는 동일하다고 보고, time 크기만 확인
    max_time = 0
    freq_dim = 0
    for (x, y, info, idx) in batch_list:
        _, f, t = x.shape
        freq_dim = f
        if t > max_time:
            max_time = t
   
    batch_size = len(batch_list)
    
    x_batch = torch.zeros(batch_size, 9, freq_dim, max_time, dtype=torch.float)
    y_batch = torch.zeros(batch_size, max_time, dtype=torch.float) + -1  # -1로 padding
    
    idx_list = []
    info_list = []
    
    for i, (x, y, info, idx) in enumerate(batch_list):
        c, f, t = x.shape
        x_batch[i, :, :, :t] = x
        y_batch[i, :t] = y
        idx_list.append(idx)
        info_list.append(info)
    
    idx_tensor = torch.LongTensor(idx_list)
    
    return x_batch, y_batch, info_list, idx_tensor


class SpecArousalDataset(Dataset):
    """
    각 pickle 파일에는
      - 'x': shape (9, freq, time)
      - 'y': shape (time,)
    """
    def __init__(self, file_paths, normalize=False):
        super().__init__()
        self.file_paths = file_paths
        self.normalize = normalize
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        x = data_dict['x']  # shape: (9, freq, time)
        y = data_dict['y']  # shape: (time,)

        info = {
            'freqs': data_dict['freqs'],
            'times': data_dict['times'],
            'y_time': data_dict['y_time'],
            'total_samples': len(data_dict['y_time']) 
        }

        # numpy -> torch
        x = torch.from_numpy(x)  # (9, freq, time)
        y = torch.from_numpy(y)  # (time,)

        # # Normalize spectrogram
        if self.normalize:
            x = (x - x.mean()) / x.std()

        return x, y, info, idx


class BatchArousalDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, label_list, transforms = None):
        super().__init__()

        self.data_list = data_list
        self.label_list = label_list
        self.transforms = transforms

    def __len__(self):
        return len(self.label_list)        
        
    def __getitem__(self, idx):
        x, y = self.data_list[idx], self.label_list[idx]
        
        if self.transforms is not None:
            x, y = self.transforms(x, y)
        
        return x, y, idx
    
class OnTheFlyArousalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_list, num_channels, transforms = None, eval=False):
        super().__init__()

        self.data_dir = data_dir
        self.file_list = file_list
        self.num_channels = num_channels
        self.transforms = transforms
        self.eval = eval
        self.cache = {}

    def __len__(self):
        return len(self.file_list)        
        
    def __getitem__(self, idx):
        if self.eval and idx in self.cache:
            x, y = self.cache[idx]
        else:
            x, y = self.load_labeled_data(self.file_list[idx])

        if self.eval and idx not in self.cache:
            self.cache[idx] = (x, y)
        
        if self.transforms is not None:
            x, y = self.transforms(x, y)
        
        return x, y, idx

    def load_labeled_data(self, filename):
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            d = pickle.load(f)
            x, y = d['x'], d['y'].astype(np.int64)
            if self.num_channels != 9:
                x = x[:self.num_channels,:]
            
        return x, y
    

class MultiTaskOnTheFlyArousalDataset(torch.utils.data.Dataset):
    def __init__(self, arousal_dir, sleep_dir, file_list, num_channels, transforms = None):
        super().__init__()

        self.arousal_dir = arousal_dir
        self.sleep_dir = sleep_dir
        self.file_list = file_list
        self.num_channels = num_channels
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)        
        
    def __getitem__(self, idx):
        x, arousal_y, sleep_y = self.load_labeled_data(self.file_list[idx])
        
        if self.transforms is not None:
            x, arousal_y = self.transforms(x, arousal_y)
        
        return x, arousal_y, sleep_y, idx

    def load_labeled_data(self, filename):
        with open(os.path.join(self.arousal_dir, filename), 'rb') as f:
            d = pickle.load(f)
            x, arousal_y = d['x'], d['y'].astype(np.int64)
            if self.num_channels != 9:
                x = x[:self.num_channels,:]

        with open(os.path.join(self.sleep_dir, filename), 'rb') as f:
            d = pickle.load(f)
            sleep_y = d['y'].astype(np.int64)
            
        return x, arousal_y, sleep_y