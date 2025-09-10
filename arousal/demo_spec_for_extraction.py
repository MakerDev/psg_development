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
from common.eval_utils import event_level_analysis
from utils.tools import load_edf_file, save_arousal_xml, load_edf_only
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
            'y_sleep_time': data_dict['y_sleep_time'],
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
    
    th = float(pretrained_path.split('_')[-1].replace('.pt', '').replace('th', ''))
    th = 0.3127


    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
      
    y_pred, y_target, y_prob, y_sleep, acc, precision, recall, fl = eval_fn2(model, val_loader, device, th=th)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")
    
    if save_path is not None:
        excel_path = save_path + "/" + edf_name.replace(".edf", "_event_comparison.xlsx")
    else:
        excel_path = None

    y_pred = postprocess_arousal_preds(y_pred, min_len=3.8, fs=50)
    print("--After Postprocessing--")

    if excel_path is not None:
        event_level_analysis(y_pred, y_target, y_prob, excel_path, overlap_th=0.1, y_sleep=y_sleep)
        stats = None
    else:
        stats = event_level_analysis(y_pred, y_target, y_prob, None, overlap_th=0.1, return_stats=True, y_sleep=y_sleep)
    acc, precision, recall, fl = accuracy_score(y_target, y_pred), precision_score(y_target, y_pred), recall_score(y_target, y_pred), f1_score(y_target, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")

    if save_path is not None:
        save_path = save_path + "/" + edf_name.replace(".edf", "_AROUS_PRED.xml")
        save_to_xml(edf_path, y_pred, save_path)
        print(f"Saved XML at: {save_path}")
    
    return acc, precision, recall, fl, stats

if __name__ == "__main__":
    # edf_dir = "/home/honeynaps/data/GOLDEN/EDF2"
    # edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]
    # edf_files = [f for f in edf_files if "SCH_M_20_OV_230111R1_NO" not in f]
    # edf_path = os.path.join(edf_dir, edf_files[3])
    # save_path = '/home/honeynaps/data/shared/arousal'
    # main(edf_path, save_path)
    # exit(0)

    # edf_dir = "/home/honeynaps/data/HN_DATA_AS/EDF"
    # edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    edf_dir = "/home/honeynaps/data/GOLDEN/EDF2"
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]
    edf_files = [f for f in edf_files if "SCH_M_20_OV_230111R1_NO" not in f]

    stats_header = ["edf_name",
                    "n_events_found", 
                    "n_events_missed",
                    "n_events_unmatched",
                    "detection_ratio",
                    "mean_overlap_ratio",
                    "avg_front_overhang",
                    "avg_back_overhang",
                    "avg_front_underhang",
                    "avg_back_underhang",
                    "matched_pred_ratio",
                    "acc", "precision", "recall", "f1"]
    stat_lines = [stats_header]
    avg_acc, avg_precision, avg_recall, avg_f1 = 0, 0, 0, 0
    for edf_file in edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        print("\nProcessing", edf_path)
        acc, precision, recall, f1, stats = main(edf_path, None)
        avg_acc += acc
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1

        # print("mean_overlap_ratio", stats['mean_overlap_ratio'], "Avg_front_overhang", stats['avg_front_overhang'], "Avg_back_overhang", stats['avg_back_overhang'])
        # print("Avg_front_underhang", stats['avg_front_underhang'], "Avg_back_underhang", stats['avg_back_underhang'])
        # print("Detection_ratio", stats['detection_ratio'])

        print(f"Mean Overlap Ratio: {stats['mean_overlap_ratio']:.4f}, "
              f"Avg Front Overhang: {stats['avg_front_overhang']:.4f}, "
              f"Avg Back Overhang: {stats['avg_back_overhang']:.4f}\n "
              f"Avg Front Underhang: {stats['avg_front_underhang']:.4f}, "
              f"Avg Back Underhang: {stats['avg_back_underhang']:.4f}")
        print(f"Detection Ratio: {stats['detection_ratio']:.4f}")

        stat = [edf_file] + list(stats.values()) + [acc, precision.item(), recall, f1]
        stat_lines.append(stat)

    # with pd.ExcelWriter("/home/honeynaps/data/shared/arousal/arousal_stats_spec.xlsx") as writer:
    #     df = pd.DataFrame(stat_lines[1:], columns=stat_lines[0])
    #     df.to_excel(writer, index=False)
    
    avg_acc /= len(edf_files)
    avg_precision /= len(edf_files)
    avg_recall /= len(edf_files)
    avg_f1 /= len(edf_files)
    print(f"\nAverage Accuracy: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")