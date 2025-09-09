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
from models.DeepSleepSota import DeepSleepNetSota
from models.DeepSleepSota2D import DeepSleepSota2D
from utils.eval_helper import *
from utils.tools import load_edf_file, save_arousal_xml, load_edf_only
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.transforms import build_transforms
from utils.datasets import SpecArousalDataset, spec_collate_fn
import xml.etree.ElementTree as ET
import datetime


class OnTheFlyArousalDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, num_channels, transforms = None, eval=False):
        super().__init__()

        self.file_paths = file_paths
        self.num_channels = num_channels
        self.transforms = transforms
        self.eval = eval
        self.cache = {}

    def __len__(self):
        return len(self.file_paths)        
        
    def __getitem__(self, idx):
        if self.eval and idx in self.cache:
            x, y = self.cache[idx]
        else:
            x, y = self.load_labeled_data(self.file_paths[idx])

        if self.eval and idx not in self.cache:
            self.cache[idx] = (x, y)
        
        if self.transforms is not None:
            x, y = self.transforms(x, y)
        
        return x, y, idx

    def load_labeled_data(self, file_path):
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
            x, y = d['x'].astype(np.float32), d['y'].astype(np.int64)
            if self.num_channels != 9:
                x = x[:self.num_channels,:]
            
        return x, y


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_to_xml(edf_path, y, save_path, sfreq=50, base_time=None, desc="AROUS"):
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
    save_arousal_xml(base_time, y, sfreq, save_path, min_duration=3, description=desc)


def postprocess_arousal_preds_spec(preds, min_len=5, fs=50):
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

def postprocess_arousal_preds_time(preds, fs=50, a=0.1, b=0.4, min_sec=17):
    # 3초 -> fs * 3 = 150 샘플
    min_event_samples = int(min_sec * fs)
    
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


def eval_time(model, loader, device, th=0.55):
    model.eval()
    
    with torch.no_grad():
        acc, precision, recall, f1 = 0, 0, 0, 0
        for x, y, idx in loader:          
            x = x.to(device = device)
            y = y.to(device = device)

            # with torch.amp.autocast(device, enabled = torch.cuda.is_available() and not comp_score):
            y_pred = model(x, True)
            # y_pred = torch.sigmoid(y_pred)
            for i, single_idx in enumerate(idx):
                record_name = str(single_idx.item())
                y_target = y[i].view(-1).to('cpu')
                y_pred_i = y_pred[i].view(-1).to('cpu')
                y_pad_mask = y_target != -1
                y_target = y_target[y_pad_mask]
                y_pred_i = y_pred_i[y_pad_mask]
                y_prob = y_pred_i
                y_pred_i = (y_pred_i > th).numpy().astype(int)
                # y_pred_i = postprocess_arousal_preds(y_pred_i, fs=50, a=0.1, b=0.4, min_sec=17)

                acc += accuracy_score(y_target, y_pred_i)
                precision += precision_score(y_target, y_pred_i)
                recall += recall_score(y_target, y_pred_i)
                f1 += f1_score(y_target, y_pred_i)

    return y_pred_i, y_target, y_prob, \
        acc/len(loader.dataset), precision/len(loader.dataset), recall/len(loader.dataset), f1/len(loader.dataset)


def eval_spec(model, loader, device, th=0.923):
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

    return y_pred_i, y_target, y_pred_logit_time, \
        acc/len(loader.dataset), precision/len(loader.dataset), recall/len(loader.dataset), f1/len(loader.dataset)


def parse_sleep_stages(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = []

    for ann in root.findall('annotation'):
        onset_str = ann.find('onset').text  # ex) "2023-07-15T22:13:00.000000"
        duration_str = ann.find('duration').text
        desc = ann.find('description').text
        
        # 만약 수면 단계가 모두 "SLEEP-" 라고 시작한다고 가정
        if desc and desc.startswith("SLEEP-W"):
            # onset -> datetime
            dt_obj = datetime.datetime.strptime(onset_str, "%Y-%m-%dT%H:%M:%S.%f")
            start_sec = dt_obj.timestamp()
            dur = float(duration_str)
            end_sec = start_sec + dur

            results.append({
                'start_sec': start_sec,
                'end_sec': end_sec,
                'description': desc
            })
    
    results.sort(key=lambda x: x['start_sec'])
    return results


def arousal_events_to_real_time(arousal_events, sfreq, meas_date):
    results = []
    base_epoch = meas_date.timestamp()  # float

    for (samp_s, samp_e, _) in arousal_events:
        # 시각(초) offset
        offset_s = samp_s / sfreq
        offset_e = samp_e / sfreq

        start_sec = base_epoch + offset_s
        end_sec   = base_epoch + offset_e

        results.append({
            'start_sec': start_sec,
            'end_sec': end_sec,
            'description': 'AROUSAL'
        })
    
    return results


def merge_sleep_w_intervals(sleep_stages):
    intervals = []
    for stg in sleep_stages:
        desc = stg['description']
        if desc == 'SLEEP-W':
            intervals.append((stg['start_sec'], stg['end_sec']))
    
    intervals.sort(key=lambda x: x[0])

    # 병합
    merged = []
    for iv in intervals:
        if not merged:
            merged.append(iv)
        else:
            ps,pe = merged[-1]
            cs,ce = iv
            if cs <= pe:  # 겹침
                merged[-1] = (ps, max(pe, ce))
            else:
                merged.append(iv)
    return merged

def filter_arousals_in_sleep_w(arousal_list, w_list, offset=15):
    filtered = []
    for aro in arousal_list:
        As = aro['start_sec']
        Ae = aro['end_sec']

        contained = False
        for (Ws, We) in w_list:
            if (As - offset >= Ws) and (Ae + offset <= We):
                contained = True
                break
        
        if not contained:
            filtered.append(aro)
    return filtered

def arousal_list_to_1d(final_arousals, meas_date, fs, total_samples):
    base_sec = meas_date.timestamp()
    arr = np.zeros(total_samples, dtype=int)
    for aro in final_arousals:
        s_sec = aro['start_sec']
        e_sec = aro['end_sec']
        s_samp = int(np.floor((s_sec - base_sec)*fs))
        e_samp = int(np.ceil((e_sec - base_sec)*fs))
        if s_samp<0: s_samp=0
        if e_samp>=total_samples: e_samp=total_samples-1
        if s_samp<=e_samp:
            arr[s_samp:e_samp+1]=1
    return arr

def evaluate_spec_model(edf_path, device, save_path=None):
    edf_name = os.path.basename(edf_path)
    spec_dir = "/home/honeynaps/data/GOLDEN/AROUS_SPEC/AROUSAL_SPEC_50_PAD_tight"
    val_files = [os.path.join(spec_dir, edf_name.replace(".edf", ".pkl"))]
    
    spec_val_dataset  = SpecArousalDataset(val_files)
    spec_val_loader   = DataLoader(spec_val_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=spec_collate_fn)
    spec_model = DeepSleepSota2D(in_channels=9).to(device)
    pretrained_path = "/home/honeynaps/data/saved_models_spec/2DUnet__ep14_auprc0.8036_f1_0.7604_th0.923.pt"
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW[2, 4, 6]__f1_0.8015__PAD_tight_lr0.0010_fs50_ep13_auprc0.8027_th0.3127.pt"
    th_spec = float(pretrained_path.split('_')[-1].replace('.pt', '').replace('th', ''))
    spec_model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
      
    y_pred, y_target, y_prob, acc, precision, recall, fl = eval_spec(spec_model, spec_val_loader, device, th=th_spec)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")

    y_pred = postprocess_arousal_preds_spec(y_pred, min_len=3.8, fs=50)
    print("--After Postprocessing--")

    gt_events, pred_events, n_events_found, n_events_missed, n_events_unmatched = event_level_analysis(y_pred, y_target, y_prob=y_prob, excel_path=None, overlap_th=0.1)
    acc, precision, recall, fl = accuracy_score(y_target, y_pred), precision_score(y_target, y_pred), recall_score(y_target, y_pred), f1_score(y_target, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")
    print(f"Events Found: {n_events_found}, Events Missed: {n_events_missed}, Events Unmatched: {n_events_unmatched}")
    event_detection_ratio = n_events_found / (n_events_found + n_events_missed)
    print(f"Event Detection Ratio: {event_detection_ratio:.4f}")

    with open(val_files[0], "rb") as f:
        meas_date = pickle.load(f)['meas_date']
        meas_date = meas_date.replace(tzinfo=None)

    sleep_xml_path = edf_path.replace(edf_name, f"SLEEP/{edf_name}").replace(".edf", "_SLEEP.xml").replace("EDF", "EBX")
    arousal_list = arousal_events_to_real_time(pred_events, sfreq=50, meas_date=meas_date)
    sleep_stages = parse_sleep_stages(sleep_xml_path)
    sleep_w_intervals = merge_sleep_w_intervals(sleep_stages)
    arousal_list = filter_arousals_in_sleep_w(arousal_list, sleep_w_intervals)
    y_pred = arousal_list_to_1d(arousal_list, meas_date, fs=50, total_samples=len(y_target))

    print("--After Filtering--")

    gt_events, pred_events, n_events_found, n_events_missed, n_events_unmatched = event_level_analysis(y_pred, y_target, y_prob=y_prob, excel_path=None, overlap_th=0.1)
    acc, precision, recall, fl = accuracy_score(y_target, y_pred), precision_score(y_target, y_pred), recall_score(y_target, y_pred), f1_score(y_target, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")
    print(f"Events Found: {n_events_found}, Events Missed: {n_events_missed}, Events Unmatched: {n_events_unmatched}")
    event_detection_ratio = n_events_found / (n_events_found + n_events_missed)
    print(f"Event Detection Ratio: {event_detection_ratio:.4f}")

    if save_path is not None:
        save_path = save_path + "/" + edf_name.replace(".edf", "_AROUS_SPEC.xml")
        save_to_xml(edf_path, y_pred, save_path, desc="AROUS_SPEC")
        print(f"Saved XML at: {save_path}")

    return y_pred, y_target, pred_events


def evaluate_time_model(edf_path, device, save_path=None):
    time_dir = '/home/honeynaps/data/GOLDEN/PICKLE/AROUSAL_VER2_50_PAD'
    time_dir = '/home/honeynaps/data/GOLDEN/PICKLE/AROUSAL_TIME_50'
    edf_name = os.path.basename(edf_path)
    val_files = [os.path.join(time_dir, edf_name.replace(".edf", ".pkl"))]

    if "VER2" in time_dir:
        val_transforms = ["NormaliseOnly"]
        val_transforms = build_transforms(val_transforms, n_channels = 9)
        
        time_val_dataset = OnTheFlyArousalDataset(val_files, 9, val_transforms)
    else:
        time_val_dataset = OnTheFlyArousalDataset(val_files, 9)

    val_loader  = DataLoader(time_val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)

    # model
    model = DeepSleepNetSota(n_channels=9).to(device)
    pretrained_path = "/home/honeynaps/data/shared/arousal/saved_models/deepsleep_tight_asam_0.6587.pt"
    pretrained_path = "/home/honeynaps/data/saved_models_time/TimenormalW[4, 6, 8]__f1_0.7997__lr0.0001_fs50_ep13_auprc0.8312_th0.3908.pt"
    # pretrained_path = "/home/honeynaps/data/shared/arousal/saved_models/deepsleep_loose_asam_0.55.pt"
    th = float(pretrained_path.split('_')[-1].replace('.pt', '').replace("th", ""))
    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

    y_pred, y_target, y_prob, acc, precision, recall, fl = eval_time(model, val_loader, device, th=th)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")

    tag = "_tight" if "tight" in pretrained_path else "_loose"

    # if tag == "_loose":
    #     y_pred = postprocess_arousal_preds_time(y_pred, fs=50, a=0.1, b=0.4, min_sec=17)
    # else:
    #     y_pred = postprocess_arousal_preds_time(y_pred, fs=50, a=0.0, b=0.0, min_sec=3)
    print("--After Postprocessing--")
    gt_events, pred_events, n_events_found, n_events_missed, n_events_unmatched = event_level_analysis(y_pred, y_target, y_prob=y_prob, excel_path=None, overlap_th=0.1)
    acc, precision, recall, fl = accuracy_score(y_target, y_pred), precision_score(y_target, y_pred), recall_score(y_target, y_pred), f1_score(y_target, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")
    
    print(f"Events Found: {n_events_found}, Events Missed: {n_events_missed}, Events Unmatched: {n_events_unmatched}")
    event_detection_ratio = n_events_found / (n_events_found + n_events_missed)
    print(f"Event Detection Ratio: {event_detection_ratio:.4f}")

    if save_path is not None:
        save_path = save_path + "/" + edf_name.replace(".edf", "_AROUS_TIME.xml")
        save_to_xml(edf_path, y_pred, save_path, desc="AROUS_TIME")
        print(f"Saved XML at: {save_path}")

    return y_pred, y_target, pred_events


def compute_overlap(A_pred: np.ndarray, B_pred: np.ndarray, y_true: np.ndarray):
    assert A_pred.shape == B_pred.shape == y_true.shape, "모든 배열은 같은 길이여야 합니다."
    
    # 모델 A의 지표
    A_tp = (A_pred == 1) & (y_true == 1)
    A_fp = (A_pred == 1) & (y_true == 0)
    A_fn = (A_pred == 0) & (y_true == 1)
    
    # 모델 B의 지표
    B_tp = (B_pred == 1) & (y_true == 1)
    B_fp = (B_pred == 1) & (y_true == 0)
    B_fn = (B_pred == 0) & (y_true == 1)
    
    # 겹침 계산
    overlap = {}
    for label, A_mask, B_mask in [
        ("TP", A_tp, B_tp),
        ("FP", A_fp, B_fp),
        ("FN", A_fn, B_fn),
    ]:
        # count
        cnt_A = np.sum(A_mask)
        cnt_B = np.sum(B_mask)
        cnt_overlap = np.sum(A_mask & B_mask)
        # 비율 (겹친 수 / 각 모델의 해당 지표 수)
        pct_A = cnt_overlap / cnt_A * 100 if cnt_A > 0 else 0.0
        pct_B = cnt_overlap / cnt_B * 100 if cnt_B > 0 else 0.0
        
        overlap[label] = {
            "A_count": int(cnt_A),
            "B_count": int(cnt_B),
            "overlap_count": int(cnt_overlap),
            "overlap_pct_of_A": pct_A,
            "overlap_pct_of_B": pct_B,
        }
    
    return overlap

def main(edf_path, excel_path=None, save_path=None, mode="union"):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = f'cuda:3' if torch.cuda.is_available() else 'cpu'

    print("Evaluating Spec Model...")
    spec_preds, y_target, spec_pred_events = evaluate_spec_model(edf_path, device, save_path)

    print("\nEvaluating Time Model...")
    time_preds, _, time_pred_events = evaluate_time_model(edf_path, device, save_path)

    print(compute_overlap(spec_preds, time_preds, y_target))
    y_pred, final_events = combine_two_models_events(
        spec_pred_events,
        time_pred_events,
        len(y_target),
        mode=mode
    )

    acc, precision, recall, fl = accuracy_score(y_target, spec_preds), precision_score(y_target, spec_preds), recall_score(y_target, spec_preds), f1_score(y_target, spec_preds)
    return acc, precision, recall, fl, None

    print("\n--After Combining--")
    acc, precision, recall, fl = accuracy_score(y_target, y_pred), precision_score(y_target, y_pred), recall_score(y_target, y_pred), f1_score(y_target, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")
    
    if excel_path is not None:
        excel_path = excel_path.replace(".xlsx", f"_{mode}.xlsx")    
    # gt_events, pred_events, n_events_found, n_events_missed, n_events_unmatched = event_level_analysis(y_pred, y_target, y_prob=None, excel_path=excel_path, overlap_th=0.1)
    stats = event_level_analysis(y_pred, y_target, y_prob=None, excel_path=excel_path, overlap_th=0.1, return_stats=True)
    # print(f"Events Found: {n_events_found}, Events Missed: {n_events_missed}, Events Unmatched: {n_events_unmatched}")

    if save_path is not None:
        edf_name = os.path.basename(edf_path)
        save_path = save_path + "/" + edf_name.replace(".edf", "_AROUS.xml")
        save_to_xml(edf_path, y_pred, save_path)
        print(f"Saved XML at: {save_path}")

    return acc, precision, recall, fl, stats
  

if __name__ == "__main__":
    # edf_path = "/home/honeynaps/data/GOLDEN/EDF2/SCH_F_20_OB_231128R4_NO.edf"
    # # edf_path = "/home/honeynaps/data/GOLDEN/EDF2/SCH_F_40_NW_231130R4_MO.edf"
    # save_path = '/home/honeynaps/data/shared/arousal'
    # edf_name = os.path.basename(edf_path)
    # excel_path = save_path + "/" + edf_name.replace(".edf", f"_event_comparison.xlsx")

    # main(edf_path, save_path=save_path, excel_path=excel_path)

    edf_dir = "/home/honeynaps/data/GOLDEN/EDF2"
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]
    edf_files.remove("SCH_M_20_OV_230111R1_NO.edf")

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
    mode = 'union'
    for edf_file in edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        print("\nProcessing", edf_path)
        acc, precision, recall, f1, stats = main(edf_path, None, mode=mode)
        avg_acc += acc
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1

        # stat = [edf_file] + list(stats.values()) + [acc, precision.item(), recall, f1]
        # stat_lines.append(stat)

    with pd.ExcelWriter(f"/home/honeynaps/data/shared/arousal/arousal_stats_ensemble_{mode}.xlsx") as writer:
        df = pd.DataFrame(stat_lines[1:], columns=stat_lines[0])
        df.to_excel(writer, index=False)
    
    avg_acc /= len(edf_files)
    avg_precision /= len(edf_files)
    avg_recall /= len(edf_files)
    avg_f1 /= len(edf_files)
    print(f"\nAverage Accuracy: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")