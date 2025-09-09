import torch
import numpy as np
import pickle
import os
import argparse
import random
from models.cnn_encoders import *
from utils.transforms import *
from utils.tools import *
from prep_window_wise import load_edf_for_demo, load_only_edf
from utils.post_process import run_postprocess
from datetime import datetime


def save_to_xml(edf_path, y, save_path, base_time, fs=50, probs=None):
    if base_time is None:
        raw = load_edf_file(edf_path, preload=True, resample=fs, preset="STAGENET", exclude=True, missing_ch='raise')
        base_time = raw.info['meas_date']
    else:
        base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
    save_sleepstage_xml(base_time, y, save_path, probs=probs)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, edf_path, xml_path, num_channels=9, fs=50, transforms=None, start_time=None, missing_ch='raise'):        
        self.num_channels = num_channels
        self.fs = fs
        self.transform = transforms

        self.data_list, self.label_list = [], []

        if xml_path:
            data = load_edf_for_demo(edf_path, xml_path, fs, fill_na=False, missing_ch=missing_ch)

            if isinstance(data, tuple): # Error occurred
                print(data[1])
                raise ValueError(data[1])

            x, y = data['x'], data['y'].astype(np.int64)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            self.data_list.append(x)
            self.label_list.append(y)
        else:
            X = load_only_edf(edf_path, start_time, fs, missing_ch=missing_ch)
            self.data_list.append(X)
            self.label_list.append(np.zeros(X.shape[0]))

        self.data_list = [torch.tensor(data, dtype=torch.float32) for data in self.data_list]
        self.label_list = [torch.tensor(labels, dtype=torch.long) for labels in self.label_list]
        self.data_list = torch.concat(self.data_list, dim=0).unsqueeze(1)
        self.label_list = torch.concat(self.label_list, dim=0)

        self.data_list, self.label_list = self._group_data(self.data_list, self.label_list, 1)
        self._permute_data()

    def _group_data(self, data, labels, n):
        grouped_data = []
        grouped_labels = []
        for idx in range(0, len(data) - n + 1):
            grouped_data.append(data[idx:idx+n])
            grouped_labels.append(labels[idx+n-1])  # Label for the last item in the group
        
        grouped_data = torch.stack(grouped_data)
        grouped_labels = torch.tensor(grouped_labels, dtype=torch.long)
        
        return grouped_data, grouped_labels

    def _permute_data(self):     
        self.data_list = self.data_list.reshape(-1, 1, self.data_list.size(3), self.data_list.size(4))
        self.data_list = self.data_list.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]

        if self.transform:
            original_shape = data.shape
            data = data.reshape(self.num_channels, -1)
            data, label = self.transform(data, label)
            data = data.reshape(original_shape)

        return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default='/home/honeynaps/data/250718_CND/EDF/CND-241017R1_M-40-OV-SE.edf')
    parser.add_argument('--dest', type=str, default='/home/honeynaps/data/shared')
    parser.add_argument('--start_time', type=str, default='/home/honeynaps/data/250718_CND/EBX/SLEEP/CND-241017R1_M-40-OV-SE_SLEEP.xml')
    parser.add_argument('--xml', type=str, default=None)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=5)
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

    pretrained = False
    pin_memory = True
    num_channels = args.num_channels
    missing_ch = 1
    dataset = TestDataset(args.edf, args.xml, num_channels, args.fs, transforms=transforms, start_time=args.start_time, missing_ch=missing_ch)

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

    pretrained_path = f'/home/honeynaps/data/shared/sleep_stage/saved_models/pretrained_asam_ver2.pt'   #pretrained_asam.pt'
    pretrained_path = f"/home/honeynaps/data/shared/sleep_stage/saved_models/pretrained_asam_ver3.pt"
    model.load_state_dict(torch.load(pretrained_path, map_location=device))

    num_signals = 1500 * (args.fs // 50)
    save_dir = args.dest

    y_true, y_pred = [], []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for data, labels in dataset:
            data, labels = data.to(device), labels.to(device)
            data = data.reshape(-1, num_channels, 1, num_signals)
            
            outputs = model(data)
            outputs = outputs.reshape(-1, outputs.size(-1))
            labels = labels.reshape(-1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(dim=-1).cpu().numpy().tolist())

            probs = torch.softmax(outputs, dim=-1)
            all_probs.extend(probs.cpu().numpy().tolist())

        y_pred = run_postprocess(y_pred, 6)
        
        edf_name = os.path.basename(args.edf)
        save_path = os.path.join(save_dir, edf_name.replace('.edf', '_SLEEP_PRED.xml'))
        save_to_xml(args.edf, y_pred, save_path, base_time=args.start_time, fs=args.fs, probs=all_probs)
        print(f'Saved {save_path}')

        if args.xml:
            acc = np.mean(np.array(y_pred) == np.array(y_true))
            print(f'Accuracy: {acc:.4f}')

