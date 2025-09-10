import torch
import os
import numpy as np
import torch
import random
import pickle
import datetime
import argparse
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.DeepSleepSota2D import DeepSleepSota2D
from common.eval_utils import event_level_analysis, find_predicted_events
from utils.tools import load_edf_file, save_arousal_xml, load_edf_only
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mne.filter import filter_data

def band_pass_filter(x, lf, hf, fs=100, tb='auto'):

    return filter_data(
        x, sfreq=fs, l_freq=lf, h_freq=hf, verbose=False,
        l_trans_bandwidth=tb, h_trans_bandwidth=tb)

def str2bool(v):
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
    save_arousal_xml(base_time, y, sfreq, save_path, min_duration=3, description='AROUS_PRED')


def postprocess_arousal_preds(preds, min_len=5, fs=50):
    min_event_samples = int(min_len * fs)
    
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
                    if end_idx >= start_idx:
                        new_preds[start_idx: end_idx + 1] = 1
                
                in_event = False

    return new_preds

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
            'total_samples': len(data_dict['y_time']) ,
            'y_sleep_time': data_dict['y_sleep_time'] if 'y_sleep_time' in data_dict else data_dict['y_time'], # Hack
        }

        # numpy -> torch
        x = torch.from_numpy(x)  # (9, freq, time)
        y = torch.from_numpy(y)  # (time,)

        # # Normalize spectrogram
        if self.normalize:
            x = (x - x.mean()) / x.std()

        return x, y, info, idx


def map_spec_pred_to_time(
    pred_1d,        # shape: (time_bins,) => STFT each bin의 예측값 (0~1 등)
    times,          # shape: (time_bins,) => make_spectrogram의 STFT 윈도우 중심 시각(초)
    total_samples,  # 원본 시계열 전체 샘플 수
    fs=50,          # 샘플링 레이트
    nperseg=50,     # STFT 윈도우 크기(샘플)
    mode='average'
):
    # 윈도우 중심으로부터 앞뒤 절반 길이(초 단위)
    half_win_sec = nperseg / (2.0 * fs)  # 예: 2초 윈도우라면 1초
    
    y_time = np.zeros(total_samples, dtype=np.float32)
    count  = np.zeros(total_samples, dtype=np.float32)  # 몇 개 윈도우가 겹쳤는지 기록

    time_bins = len(times)

    for i in range(time_bins):
        center_sec = times[i]       # i번째 bin 중심 시각 (초)
        start_sec = center_sec - half_win_sec
        end_sec   = center_sec + half_win_sec
        
        # 원본 샘플 인덱스로 환산
        start_idx = int(np.floor(start_sec * fs))
        end_idx   = int(np.ceil(end_sec * fs))
        
        # 유효 범위로 자르기
        if start_idx < 0:
            start_idx = 0
        if end_idx > total_samples:
            end_idx = total_samples

        if start_idx >= end_idx:
            continue
        
        if mode == 'average':
            # 해당 구간에 pred_1d[i]를 누적
            y_time[start_idx:end_idx] += pred_1d[i]
            count[start_idx:end_idx]  += 1.0
        
        elif mode == 'max':
            # 기존 값과 비교해 최댓값
            y_time[start_idx:end_idx] = np.maximum(
                y_time[start_idx:end_idx],
                pred_1d[i]
            )
        # 필요하다면 다른 방식(가중 합 등)도 가능

    if mode == 'average':
        # 겹친 구간 개수로 나눠 평균
        nonzero_mask = (count > 0)
        y_time[nonzero_mask] /= count[nonzero_mask]

    return y_time


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


def eval_fn2(model, loader, device, th=0.923):
    model.eval()
    
    with torch.no_grad():
        acc, precision, recall, f1 = 0, 0, 0, 0
        for x, y, info, idx in loader:
            x = x.to(device)
            y = y.to(device)
            
            # forward
            y_pred_2d = model(x)  # (B,1,freq,T_max), sigmoid output in forward
            # freq pooling -> (B,1,T_max)
            y_pred_1d = y_pred_2d.mean(dim=2)  # or .max(dim=2)[0]
            
            # padding mask
            pad_mask = (y != -1)
            y_pred_1d = y_pred_1d.squeeze(1)
            
            y_pred_1d[~pad_mask] = 0.0
            y[~pad_mask] = 0

            # y_pred = torch.sigmoid(y_pred)
            for i, single_idx in enumerate(idx):
                info_i = info[i]
                times = info_i['times']
                total_samples = info_i['total_samples']
                y_target = info_i['y_time']
                y_sleep = info_i['y_sleep_time'].astype(int)

                valid_idx = pad_mask[i]  # shape: (T_max,)
                # y_target = y[i][valid_idx].cpu()
                y_pred_i = y_pred_1d[i][valid_idx].cpu()
                y_pred_logit_time = map_spec_pred_to_time(y_pred_i.numpy(), times, total_samples, fs=50, nperseg=50)
                y_pred_i = (y_pred_i > th).numpy().astype(int)
                y_pred_i = map_spec_pred_to_time(y_pred_i, times, total_samples, fs=50, nperseg=50)
                y_pred_i = (y_pred_i > 0.5).astype(int)

                acc += accuracy_score(y_target, y_pred_i)
                precision += precision_score(y_target, y_pred_i)
                recall += recall_score(y_target, y_pred_i)
                f1 += f1_score(y_target, y_pred_i)

    return y_pred_i, y_target, y_pred_logit_time, y_sleep, \
        acc/len(loader.dataset), precision/len(loader.dataset), recall/len(loader.dataset), f1/len(loader.dataset)

def chunkwise_mean_rms_norm(x, fs=50, window_min=18):
    window_size = int(window_min * 60 * fs)  # 18분 * 60초 * fs
    out = np.zeros_like(x, dtype=np.float32)

    for ch_idx in range(x.shape[0]):
        ch_data = x[ch_idx]
        length = ch_data.shape[0]

        start_idx = 0
        while start_idx < length:
            end_idx = min(start_idx + window_size, length)
            chunk = ch_data[start_idx:end_idx]

            mean_val = np.mean(chunk)
            rms_val = np.sqrt(np.mean(chunk**2))
            if rms_val < 1e-12:
                rms_val = 1e-12

            out[ch_idx, start_idx:end_idx] = (chunk - mean_val) / rms_val

            start_idx = end_idx  # 오버랩 없이 다음 구간으로 이동

    return out

def prep_psg_signal(x, transpose=True, fs=50):
    # EEG (0~5)
    eeg = np.array([
        band_pass_filter(x=x[i], lf=0.5, hf=24.999, fs=fs)
        for i in range(6)
    ])  
    # EOG (6,7)
    eog = np.array([
        band_pass_filter(x=x[i], lf=0.5, hf=24.999, fs=fs)
        for i in range(6,8)
    ])  
    # EMG (8)
    emg = band_pass_filter(x=x[-1], lf=0.5, hf=24.999, fs=fs)[np.newaxis,:]

    x_filtered = np.vstack([eeg, eog, emg])

    x_norm = chunkwise_mean_rms_norm(x_filtered, fs=fs, window_min=18)

    if transpose:
        x_norm = x_norm.T

    return x_norm

def main(edf_path, save_path=None):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'

    arousal_dir = os.path.dirname(edf_path).replace("EDF", "AROUS_SPEC")
    arousal_dir = "/home/honeynaps/data/GOLDEN/AROUS_SPEC"
    
    test_dir = f"{arousal_dir}/AROUSAL_SPEC_50_PAD_tight"

    edf_name = os.path.basename(edf_path)
    val_files = [os.path.join(test_dir, edf_name.replace(".edf", ".pkl"))]

    val_dataset  = SpecArousalDataset(val_files)
    val_loader   = DataLoader(val_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              collate_fn=spec_collate_fn)

    # model
    model = DeepSleepSota2D(in_channels=9).to(device)
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW6__f1_0.7930_lr0.0010_fs50_ep14_auprc0.8137_th0.2387.pt" # 0.77
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW6__f1_0.7915__PAD_tight_lr0.0010_fs50_ep19_auprc0.8072_th0.3003.pt" # 0.7709
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW6__f1_0.7872__PAD_tight_lr0.0010_fs50_ep9_auprc0.8290_th0.3584.pt" # 0.777
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW2__f1_0.7874_lr0.0010_fs50_ep11_auprc0.8451_th0.2614.pt" # 0.773
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW2__f1_0.7961_lr0.0010_fs50_ep23_auprc0.8298_th0.1966.pt" # 0.7784
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW6__f1_0.7967_lr0.0010_fs50_ep36_auprc0.7637_th0.1504.pt" # 0.78
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW2__f1_0.8020_lr0.0010_fs50_ep26_auprc0.8228_th0.3037.pt" # 0.7829
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW2__f1_0.8007__PAD_tight_lr0.0010_fs50_ep17_auprc0.8471_th0.2412.pt" # 0.7838
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW[2, 4, 6]__f1_0.8015__PAD_tight_lr0.0010_fs50_ep13_auprc0.8027_th0.3127.pt" # 0.7868 W1, 0.7869 W2, 0.7866 W3
    # pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW[1, 2, 4]__f1_0.8142__PAD_tech_per1test_1_lr0.0010_fs50_ep16_auprc0.8013_th0.5023.pt"
    
    th = float(pretrained_path.split('_')[-1].replace('.pt', '').replace('th', ''))
    th = 0.000005

    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

    y_pred, y_target, y_prob, y_sleep, acc, precision, recall, fl = eval_fn2(model, val_loader, device, th=th)
    y_pred = postprocess_arousal_preds(y_pred, min_len=3.8, fs=50)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")

    fp_events, _, tp_events = find_predicted_events(y_pred, y_target)
    raw = load_edf_file(path=edf_path, preload=True, resample=50, preset="STAGENET")
    data = raw.get_data()
    x = prep_psg_signal(data, transpose=True, fs=50)  # (time, ch)
    x = x.T  # (ch, time)

    x_time, y_time, y = [], [], []
    for fp_event in fp_events:
        start, end = fp_event
        x_time.append(np.array(x[:, start:end]))
        y_time.append(np.array(y_target[start:end]))
        y.append(0)
    for tp_event in tp_events:
        start, end = tp_event
        x_time.append(np.array(x[:, start:end]))
        y_time.append(np.array(y_target[start:end]))
        y.append(1)
        
    y = np.array(y)
    save_dir = os.path.join(arousal_dir, "events")
    save_path = os.path.join(save_dir, edf_name.replace(".edf", ".pkl"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_path, "wb") as f:
        pickle.dump({
            "x": x_time,
            "y": y,
            "y_time": y_time,
        }, f)


    return acc, precision, recall, fl

if __name__ == "__main__":
    edf_dir = "/home/honeynaps/data/GOLDEN/EDF2"
    # edf_dir = "/home/honeynaps/data/GOLDEN2/EDF"
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]
    edf_files = [f for f in edf_files if "SCH_M_20_OV_230111R1_NO" not in f]


    avg_acc, avg_precision, avg_recall, avg_f1 = 0, 0, 0, 0
    for edf_file in edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        print("\nProcessing", edf_path)
        acc, precision, recall, f1 = main(edf_path, None)
        avg_acc += acc
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1
