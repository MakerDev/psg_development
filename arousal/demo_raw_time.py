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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.DeepSleepSota import DeepSleepNetSota
from common.eval_utils import event_level_analysis
from utils.tools import load_edf_file, save_arousal_xml, load_edf_only
from utils.transforms import build_transforms



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
    save_arousal_xml(base_time, y, sfreq, save_path, min_duration=3)


def postprocess_arousal_preds(preds, fs=50, a=0.1, b=0.4, min_sec=17):
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
        x = data_dict['x'] 
        y = data_dict['y'] 

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
            x, y = d['x'], d['y'].astype(np.int64)
            if self.num_channels != 9:
                x = x[:self.num_channels,:]
            
        return x, y


def eval_fn2(model, loader, device, th=0.55):
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



def main(edf_path, save_path=None, export_xlsx=False):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'

    test_dir = '/home/honeynaps/data/GOLDEN/PICKLE/AROUSAL_VER2_50_PAD'

    edf_name = os.path.basename(edf_path)
    val_files = [os.path.join(test_dir, edf_name.replace(".edf", ".pkl"))]

    val_transforms = ["NormaliseOnly"]
    val_transforms = build_transforms(val_transforms, n_channels = 9)
    
    val_dataset = OnTheFlyArousalDataset(val_files, 9, val_transforms)
    val_loader   = DataLoader(val_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1)

    model = DeepSleepNetSota(n_channels=9).to(device)
    pretrained_path = "/home/honeynaps/data/shared/arousal/saved_models/deepsleep_tight_asam_0.6587.pt"
    # pretrained_path = "/home/honeynaps/data/shared/arousal/saved_models/deepsleep_loose_asam_0.55.pt"
    th = float(pretrained_path.split('_')[-1].replace('.pt', '').replace('th', ''))

    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
      
    y_pred, y_target, y_prob, acc, precision, recall, fl = eval_fn2(model, val_loader, device, th=th)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")

    tag = "_tight" if "tight" in pretrained_path else "_loose"

    if tag == "_loose":
        y_pred = postprocess_arousal_preds(y_pred, fs=50, a=0.1, b=0.4, min_sec=17)
    else:
        y_pred = postprocess_arousal_preds(y_pred, fs=50, a=0.0, b=0.0, min_sec=3)
    print("--After Postprocessing--")

    if export_xlsx:
        excel_path = save_path + "/" + edf_name.replace(".edf", f"_event_comparison{tag}.xlsx")
        stats = event_level_analysis(y_pred, y_target, y_prob, excel_path, overlap_th=0.1, return_stats=False)
    else:
        stats = event_level_analysis(y_pred, y_target, y_prob, None, overlap_th=0.1, return_stats=True)

    acc, precision, recall, fl = accuracy_score(y_target, y_pred), precision_score(y_target, y_pred), recall_score(y_target, y_pred), f1_score(y_target, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")

    if save_path is not None:
        save_path = save_path + "/" + edf_name.replace(".edf", "_AROUS.xml")
        save_to_xml(edf_path, y_pred, save_path)
        print(f"Saved XML at: {save_path}")
    
    return acc, precision, recall, fl, stats

if __name__ == "__main__":
    # edf_path = "/home/honeynaps/data/GOLDEN/EDF2/SCH_F_20_OB_231128R4_NO.edf"
    # save_path = '/home/honeynaps/data/shared/arousal'
    # main(edf_path, save_path, True)

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
    for edf_file in edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        print("\nProcessing", edf_path)
        acc, precision, recall, f1, stats = main(edf_path, None)
        avg_acc += acc
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1

        stat = [edf_file] + list(stats.values()) + [acc, precision.item(), recall, f1]
        stat_lines.append(stat)

    with pd.ExcelWriter("/home/honeynaps/data/shared/arousal/arousal_stats_tight_asam.xlsx") as writer:
        df = pd.DataFrame(stat_lines[1:], columns=stat_lines[0])
        df.to_excel(writer, index=False)
    
    avg_acc /= len(edf_files)
    avg_precision /= len(edf_files)
    avg_recall /= len(edf_files)
    avg_f1 /= len(edf_files)
    print(f"\nAverage Accuracy: {avg_acc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")