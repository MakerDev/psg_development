import os
import argparse
import pickle
import natsort
import numpy as np
import torch
import torch.nn as nn

from utils.transforms import build_transforms
from common.seed import set_seed
from utils.tools import load_edf_file, save_arousal_xml, load_edf_only

from models.DeepSleepNet2 import DeepSleepNet2
from models.DeepSleepSota import DeepSleepNetSota
from prep_arousal_ver3 import moving_window_mean_rms_norm
from prep_arousal_ver2 import prep_psg_signal
from datetime import datetime
from common.eval_utils import event_level_analysis

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

def save_to_xml(edf_path, y, save_path, sfreq=50, base_time=None):
    if base_time is None:
        raw = load_edf_file(
            edf_path, 
            preload=True, 
            resample=100, 
            preset="STAGENET", 
            exclude=True, 
            missing_ch='raise'
        )
        base_time = raw.info['meas_date']
    else:
        base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
    save_arousal_xml(base_time, y, sfreq, save_path, min_duration=3)


def postprocess_arousal_preds(preds, fs=50, a=0.1, b=0.4):
    # 3초 -> fs * 3 = 150 샘플
    min_event_samples = int(3 * fs)
    
    # 결과를 저장할 새로운 preds (모두 0으로 초기화)
    new_preds = np.zeros_like(preds, dtype=int)
    
    in_event = False
    start_idx = 0
    length = len(preds)

    for i in range(length):
        if not in_event:
            # 이벤트가 시작되지 않은 상태에서 1을 만나면 이벤트 시작
            if preds[i] == 1:
                in_event = True
                start_idx = i
        else:
            # 이미 이벤트 중이었고, 현재 0이거나 마지막 인덱스면 이벤트가 끝났다고 판단
            if preds[i] == 0 or i == length - 1:
                # 종료 지점 계산
                if preds[i] == 0:
                    end_idx = i - 1
                else:
                    end_idx = i  # 마지막 인덱스까지 1이었다면 i가 이벤트 끝
                
                # 이벤트 길이
                event_len = end_idx - start_idx + 1
                
                if event_len >= min_event_samples:
                    front_cut = int(event_len * a)
                    back_cut  = int(event_len * b)
                    shrunk_start = start_idx + front_cut
                    shrunk_end   = end_idx - back_cut
                    
                    if shrunk_end >= shrunk_start:
                        new_preds[shrunk_start: shrunk_end + 1] = 1
                
                in_event = False

    return new_preds

class ArousalDataset(torch.utils.data.Dataset):
    def __init__(self, edf_path, start_time, data_prep_fn, num_channels, fs=50, transforms = None):
        super().__init__()

        self.edf_path = edf_path
        self.start_time = start_time
        self.num_channels = num_channels
        self.transforms = transforms
        self.fs = fs
        self.data_prep_fn = data_prep_fn

    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        x, y = self.load_data(self.edf_path)
        
        if self.transforms is not None:
            x, y = self.transforms(x, y)

        edf_name = os.path.basename(self.edf_path)
        
        return edf_name, x, y

    def load_data(self, edf_path):
        prep_args = {
            "fs": self.fs,
        }
        
        x, y = load_edf_only(edf_path, self.data_prep_fn, self.start_time, sfreq=self.fs, prep_fn_args=prep_args)

        return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default='/home/honeynaps/data/GOLDEN/EDF2/SCH_F_20_OB_231128R4_NO.edf')
    parser.add_argument('--dest', type=str, default='/home/honeynaps/data/shared/arousal')
    parser.add_argument('--start_time', type=str, default=None, help='Start time in format "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = args.dest  #"/home/honeynaps/data/shared/arousal"

    transforms = ["NormaliseOnly"]
    transforms = build_transforms(transforms, n_channels=args.num_channels)

    prep_fn = moving_window_mean_rms_norm #if args.ver == 3 else prep_psg_signal

    dataset = ArousalDataset(args.edf, args.start_time,
                             data_prep_fn=moving_window_mean_rms_norm, 
                             num_channels=args.num_channels, fs=args.fs,
                             transforms=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = DeepSleepNetSota(n_channels=args.num_channels)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    pretrained_path = '/home/honeynaps/data/shared/arousal/saved_models/deepsleep_loose_asam_0.55.pt'   #deepsleep_tight_0.5382.pt'
    threshold = float(pretrained_path.split('_')[-1].replace('.pt', ''))

    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    model.eval()
    with torch.no_grad():
        for edf_name, data, label in loader:
            data = data.to(device) # batch size 1
            logits = model(data, True)
            preds = (logits > threshold).cpu().numpy().astype(int)
            preds = preds.squeeze()
            label = label.squeeze()

            pad_mask = label != -1
            logits = logits[pad_mask]
            label = label[pad_mask]
            preds = preds[pad_mask]

            preds = postprocess_arousal_preds(preds, fs=args.fs)

            xml_name = edf_name[0].replace('.edf', '_AROUS.xml')
            edf_path = args.edf
            save_path = os.path.join(save_dir, xml_name)

            save_to_xml(edf_path, preds, save_path, args.fs, base_time=args.start_time)
            print(f'Saved XML at: {save_path}')
