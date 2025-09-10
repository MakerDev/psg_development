import os
import argparse
import pickle
import natsort
import numpy as np
import torch
import torch.nn as nn

from utils.transforms import build_transforms
from utils.tools import load_edf_file, save_arousal_xml
from models import DeepSleepNetSota, DeepSleepNet2
from common.seed import set_seed

class OnTheFlyTestArousalDataset(torch.utils.data.Dataset):
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

        edf_name = self.file_list[idx].split('.')[0] + '.edf'
        
        return edf_name, x, y

    def load_labeled_data(self, filename):
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            d = pickle.load(f)
            x, y = d['x'], d['y'].astype(np.int64)
            if self.num_channels != 9:
                x = x[:self.num_channels,:]
            
        return x, y


def str2bool(v):
    """문자열 형태의 인자를 bool 값으로 변환하기 위한 헬퍼 함수"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_to_xml(edf_path, y, save_path, sfreq=100):
    raw = load_edf_file(
        edf_path, 
        preload=True, 
        resample=100, 
        preset="STAGENET", 
        exclude=True, 
        missing_ch='raise'
    )
    base_time = raw.info['meas_date']
    save_arousal_xml(base_time, y, sfreq, save_path)


def remove_short_events(labels, min_length=3):
    labels = labels.copy()
    start = -1
    for i in range(len(labels)):
        if labels[i] == 1 and start == -1:
            start = i
        elif labels[i] == 0 and start != -1:
            end = i - 1
            length = end - start + 1
            if length < min_length:
                labels[start:end+1] = 0
            start = -1

    if start != -1:
        end = len(labels)-1
        length = end - start + 1
        if length < min_length:
            labels[start:end+1] = 0
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deepsleepsota', help='모델 아키텍처 이름')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--nofill', type=str2bool, default=True)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)

    edf_dir = "/home/honeynaps/data/dataset2/EDF"
    save_dir = "/home/honeynaps/data/dataset2/EBX/AROUS_PRED"

    dataset_dir = f'/home/honeynaps/data/dataset2/PICKLE/AROUSAL_VER3_{args.fs}_PAD'

    file_names = natsort.natsorted(os.listdir(dataset_dir))
    random_indices = np.random.permutation(len(file_names))
    file_names = [file_names[i] for i in random_indices]
    test_files = file_names[int(args.split_ratio * len(file_names)):]

    transforms = ["NormaliseAndAddRandNoise"]
    transforms = build_transforms(transforms, n_channels=args.num_channels)

    dataset = OnTheFlyTestArousalDataset(dataset_dir, 
                                        test_files, 
                                        args.num_channels, 
                                     transforms)

    # 모델 아키텍처 선택
    if args.model == 'deepsleep2':
        model = DeepSleepNet2(in_channels=args.num_channels)
    else:
        model = DeepSleepNetSota(n_channels=args.num_channels)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # (예시) 사전학습 가중치 로드 (주석 처리)
    pretrained_path = '/home/honeynaps/data/eis/arousalnet_r1/saved_models/0.8668_th0.7025_DeepSleepSota_FS50_asl_CH9_BS2_RandShuffle_NormaliseAndAddRandNoise_LR0.0001_MIXFalse_VER3_NS_0.5_0.8_val_norm_and_noise_ep83.pt'
    model.load_state_dict(torch.load(pretrained_path))

    threshold = 0.7025 # 모델에 따라 최적값이 다름.
    model.eval()
    with torch.no_grad():
        for edf_name, data, label in dataset:
            data = data.to(device).unsqueeze(0) # batch size 1
            logits = model(data, True)
            preds = (torch.sigmoid(logits) > threshold).cpu().numpy().astype(int)
            preds = preds.squeeze()
            preds = remove_short_events(preds, min_length=3)

            # Remove padding
            pad_mask = label != -1
            label = label[pad_mask]
            preds = preds[pad_mask]
            
            xml_name = edf_name.replace('.edf', '.xml')
            edf_path = os.path.join(edf_dir, edf_name)
            save_path = os.path.join(save_dir, xml_name)
            save_to_xml(edf_path, label, save_path, args.fs)
            print(f'Saved XML at: {save_path}')
