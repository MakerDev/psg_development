import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import natsort
import argparse
import random
import datetime
import torch.utils.tensorboard as tb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from models.cnn_encoders import *
from utils.transforms import *
from utils.tools import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_to_xml(edf_path, y, save_path):
    raw = load_edf_file(edf_path, preload=True, resample=100, preset="STAGENET", exclude=True, missing_ch='raise')
    base_time = raw.info['meas_date']
    save_sleepstage_xml(base_time, y, save_path)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, file_dir, file_names, num_channels=9, fs=50, transforms=None):        
        self.num_channels = num_channels
        self.fs = fs
        self.transform = transforms

        self.edf_list, self.data_list, self.label_list = [], [], []
        for file_name in file_names:
            edf_name = file_name.split('.')[0] + '.edf'
            self.edf_list.append(edf_name)
            x, y = self.load_data(os.path.join(file_dir, file_name))
            self.data_list.append(x)
            self.label_list.append(y)

        self.data_list = [torch.tensor(data, dtype=torch.float32) for data in self.data_list]
        self.label_list = [torch.tensor(labels, dtype=torch.long) for labels in self.label_list]

    def _group_data(self, data, labels, n):
        grouped_data = []
        grouped_labels = []
        for idx in range(0, len(data) - n + 1):
            grouped_data.append(data[idx:idx+n]) 
            grouped_labels.append(labels[idx+n-1])  # Label for the last item in the group
        
        grouped_data = torch.stack(grouped_data)
        grouped_labels = torch.tensor(grouped_labels, dtype=torch.long)
        
        return grouped_data, grouped_labels

    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
            x, y = d['x'], d['y'].astype(np.int64)
            if self.num_channels != 9:
                x = x[:, :, :self.num_channels]            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
        return x, y

    def _permute_data(self, data):     
        data = data.reshape(-1, 1, data.size(3), data.size(4))
        data = data.permute(0, 3, 1, 2)
        return data

    def __len__(self):
        return len(self.sample_lens)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        file_name = self.edf_list[idx]

        data = data.unsqueeze(1)
        data, label = self._group_data(data, label, 1)
        data = self._permute_data(data)

        if self.transform:
            original_shape = data.shape
            data = data.reshape(self.num_channels, -1)
            data, label = self.transform(data, label)
            data = data.reshape(original_shape)

        return file_name, data, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--nofill', type=str2bool, default=True)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    transforms = ["NormaliseOnly"]
    transforms = build_transforms(transforms, n_channels=args.num_channels)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    edf_dir = "/home/honeynaps/data/dataset/EDF"
    save_dir = "/home/honeynaps/data/dataset/EBX/SLEEP_PRED"

    dataset_dir = f'/home/honeynaps/data/dataset/PICKLE/SLEEP_{args.fs}'
    if args.nofill:
        dataset_dir += '_NOFILL'

    file_names = natsort.natsorted(os.listdir(dataset_dir))
    random_indices = np.random.permutation(len(file_names))
    file_names = [file_names[i] for i in random_indices]
    file_names = file_names[int(args.split_ratio*len(file_names)):]

    pretrained = True
    pin_memory = True
    num_channels = args.num_channels
    dataset = TestDataset(dataset_dir, file_names, num_channels, args.fs, transforms=transforms)

    # 원하는 모델 아키텍쳐로 초기화 후 pretrained 모델 불러오기.
    if args.model == 'resnet18':
        model = resnet18(num_channels=num_channels, pretrained=pretrained)
    elif args.model == 'resnet50':
        model = resnet50(num_channels=num_channels, pretrained=pretrained)
    elif args.model == 'regnet128':
        model = regnet128(num_channels=num_channels, pretrained=pretrained)
    elif args.model == 'regnet16':
        model = regnet16(num_channels=num_channels, pretrained=pretrained)
    elif args.model == 'swin':
        model = swin_transformer(num_channels=num_channels)
    elif args.model == 'convnext':
        model = conv_next(num_channels=num_channels)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # pretrained_path = f'/home/honeynaps/data/models/saved_model.pt'
    # model.load_state_dict(torch.load(pretrained_path))

    num_signals = 1500 * (args.fs // 50)

    with torch.no_grad():
        model.eval()

        for edf_name, data, labels in dataset:
            data, labels = data.to(device), labels.to(device)
            data = data.reshape(-1, num_channels, 1, num_signals)
            
            outputs = model(data)
            outputs = outputs.reshape(-1, outputs.size(-1))
            labels = labels.reshape(-1)

            preds = outputs.argmax(dim=-1).cpu().numpy()

            save_path = os.path.join(save_dir, edf_name.replace('.edf', '.xml'))
            save_to_xml(os.path.join(edf_dir, edf_name), preds, save_path)
            print(f'Saved {save_path}')





