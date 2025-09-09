import torch
import os
import numpy as np
import random
import pickle
import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from models.DeepSleepSota2D import DeepSleepSota2D
from utils.eval_helper import event_level_analysis
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
    """
    특정 길이(min_len) 미만인 이벤트는 제거하는 후처리 함수.
    """
    min_event_samples = int(min_len * fs)
    
    new_preds = np.zeros_like(preds, dtype=int)
    
    in_event = False
    start_idx = 0
    length = len(preds)

    for i in range(length):
        if not in_event:
            if preds[i] == 1:
                in_event = True
                start_idx = i
        else:
            if preds[i] == 0 or i == length - 1:
                if preds[i] == 0:
                    end_idx = i - 1
                else:
                    end_idx = i
                event_len = end_idx - start_idx + 1
                
                if event_len >= min_event_samples:
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
            'total_samples': len(data_dict['y_time'])
        }

        x = torch.from_numpy(x)  # (9, freq, time)
        y = torch.from_numpy(y)  # (time,)

        if self.normalize:
            x = (x - x.mean()) / x.std()

        return x, y, info, idx

def map_spec_pred_to_time(
    pred_1d,        
    times,          
    total_samples,  
    fs=50,          
    nperseg=50,     
    mode='average'
):
    half_win_sec = nperseg / (2.0 * fs)  # 예: 2초 윈도우라면 중심으로 +/-1초
    
    y_time = np.zeros(total_samples, dtype=np.float32)
    count  = np.zeros(total_samples, dtype=np.float32) 

    time_bins = len(times)
    for i in range(time_bins):
        center_sec = times[i]
        start_sec = center_sec - half_win_sec
        end_sec   = center_sec + half_win_sec
        
        start_idx = int(np.floor(start_sec * fs))
        end_idx   = int(np.ceil(end_sec * fs))
        
        if start_idx < 0:
            start_idx = 0
        if end_idx > total_samples:
            end_idx = total_samples

        if start_idx >= end_idx:
            continue
        
        if mode == 'average':
            y_time[start_idx:end_idx] += pred_1d[i]
            count[start_idx:end_idx]  += 1.0
        elif mode == 'max':
            y_time[start_idx:end_idx] = np.maximum(y_time[start_idx:end_idx], pred_1d[i])

    if mode == 'average':
        nonzero_mask = (count > 0)
        y_time[nonzero_mask] /= count[nonzero_mask]

    return y_time

def spec_collate_fn(batch_list):
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
    all_y_pred = []
    all_y_target = []
    all_y_prob = []

    acc, precision, recall, f1 = 0, 0, 0, 0
    n_data = 0

    with torch.no_grad():
        for x, y, info, idx in loader:
            x = x.to(device)
            y = y.to(device)
            
            # forward
            y_pred_2d = model(x)  # (B,1,freq,T_max), 이미 forward에서 sigmoid라면 수정
            # freq pooling -> (B,1,T_max)
            y_pred_1d = y_pred_2d.mean(dim=2)  # or .max(dim=2)[0]
            
            # padding mask
            pad_mask = (y != -1)
            y_pred_1d = y_pred_1d.squeeze(1)
            
            # -1 부분은 실제 데이터가 없으므로 0 처리
            y_pred_1d[~pad_mask] = 0.0
            y[~pad_mask] = 0

            for i, single_idx in enumerate(idx):
                info_i = info[i]
                times = info_i['times']
                total_samples = info_i['total_samples']
                y_target = info_i['y_time']  # (time,) binary

                valid_idx = pad_mask[i]  
                y_pred_i = y_pred_1d[i][valid_idx].cpu().numpy()

                # 시계열 전체에 매핑
                y_pred_logit_time = map_spec_pred_to_time(
                    y_pred_i, times, total_samples, fs=50, nperseg=50
                )
                # threshold
                y_pred_bin = (y_pred_i > th).astype(int)
                y_pred_bin = map_spec_pred_to_time(y_pred_bin, times, total_samples, fs=50, nperseg=50)
                y_pred_bin = (y_pred_bin > 0.5).astype(int)

                acc += accuracy_score(y_target, y_pred_bin)
                precision += precision_score(y_target, y_pred_bin, zero_division=0)
                recall += recall_score(y_target, y_pred_bin, zero_division=0)
                f1 += f1_score(y_target, y_pred_bin, zero_division=0)
                n_data += 1

                # 누적 저장(이후 이벤트 레벨 분석용)
                all_y_pred.append(y_pred_bin)
                all_y_target.append(y_target)
                all_y_prob.append(
                    y_pred_logit_time  # 시계열 점수
                )

    if n_data > 0:
        acc /= n_data
        precision /= n_data
        recall /= n_data
        f1 /= n_data

    return all_y_pred, all_y_target, all_y_prob, acc, precision, recall, f1

# -------------------- Grad-CAM 부분 추가 --------------------
class GradCamHook:
    """
    특정 레이어의 forward feature map과 backward gradient를 hook으로 가져오기 위한 클래스
    """
    def __init__(self, module):
        self.module = module
        self.hook_f = None
        self.hook_b = None
        self.grad = None
        self.activation = None

    def _forward_hook(self, module, input, output):
        self.activation = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.grad = grad_out[0]

    def register(self):
        self.hook_f = self.module.register_forward_hook(self._forward_hook)
        self.hook_b = self.module.register_backward_hook(self._backward_hook)

    def remove(self):
        self.hook_f.remove()
        self.hook_b.remove()

def generate_gradcam(model, x, target_layer, device, target_idx=None):
    """
    모델에 x를 입력하고, target_layer에 대해 Grad-CAM heatmap을 계산하여 리턴.
    
    - x: (1, 9, freq, time) 형태의 입력(배치=1 가정).
    - target_idx: backward를 수행할 로짓(출력)의 인덱스(스칼라). 
                  여기서는 2D 출력 y_pred_2d가 (1,1,freq,time)이므로,
                  원하는 position에 대한 gradient를 구하거나, 
                  혹은 전체 채널/타임별로 합을 구해 backward할 수도 있음.
    """
    # GradCamHook 등록
    hook = GradCamHook(target_layer)
    hook.register()

    model.eval()
    x = x.to(device)
    # forward
    output_2d = model(x)  # (1,1,freq,T) 라고 가정
    # 여기서는 간단하게 전체 스코어 합을 대상 함수로 사용 (예: output.sum())
    # 혹은 특정 시간축에서의 score만 골라볼 수도 있음
    if target_idx is None:
        # 전체 출력의 합
        score = output_2d.sum()
    else:
        # 특정 위치(예: 특정 freq, 특정 time)의 로짓만
        score = output_2d[0, 0, target_idx[0], target_idx[1]]  # 예시
    
    # backward
    model.zero_grad()
    score.backward()

    # feature map과 gradient 가져오기
    activation = hook.activation  # shape: (1, channel, freq_map, time_map)
    grad = hook.grad             # shape: (1, channel, freq_map, time_map)

    # 채널 단위 global average pooling
    alpha = grad.view(grad.size(0), grad.size(1), -1).mean(dim=2, keepdim=True)  # (1, channel, 1)
    # (1, channel, freq_map*time_map) -> (1, channel, 1)

    # 가중치 alpha와 feature map 곱
    # activation: (1, channel, freq_map, time_map)
    # alpha:      (1, channel, 1)
    # => 채널별로 곱한 뒤 합산
    gradcam = (activation * alpha.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (1,1,freq_map,time_map)

    # ReLU
    gradcam = F.relu(gradcam)

    # 정규화 (시각화를 위해 0~1 범위로)
    gradcam = gradcam.squeeze(0).squeeze(0)  # (freq_map, time_map)
    if gradcam.max() != 0:
        gradcam /= gradcam.max()

    # hook 제거
    hook.remove()

    # shape: (freq_map, time_map)
    return gradcam.detach().cpu().numpy()

def visualize_gradcam(model, loader, device, target_layer):
    """
    val_loader에서 배치를 하나 꺼낸 뒤, Grad-CAM heatmap을 구해
    9채널의 스펙트럼 각각에 heatmap을 오버레이하여 시각화.
    """
    # 배치 1개만 가져오기
    x_batch, y_batch, info_list, idx_list = next(iter(loader))
    # x_batch shape: (B, 9, freq, time)
    x_single = x_batch[0:1]  # 첫 번째 샘플만
    info = info_list[0]

    # Grad-CAM 계산
    gradcam_map = generate_gradcam(model, x_single, target_layer, device)

    # gradcam_map의 해상도와 실제 입력 해상도가 다를 수 있으므로 보간
    # x_single.shape => (1, 9, freq, time)
    freq_in = x_single.shape[2]
    time_in = x_single.shape[3]
    freq_cam = gradcam_map.shape[0]
    time_cam = gradcam_map.shape[1]

    if (freq_in != freq_cam) or (time_in != time_cam):
        gradcam_tensor = torch.from_numpy(gradcam_map).unsqueeze(0).unsqueeze(0)  # (1,1,freq_cam,time_cam)
        gradcam_resized = F.interpolate(gradcam_tensor, size=(freq_in, time_in), mode='bilinear', align_corners=False)
        gradcam_map = gradcam_resized.squeeze().numpy()  # (freq_in, time_in)

    # 원본 스펙트럼 (9, freq, time)
    spec_9ch = x_single[0].cpu().numpy()

    # 시각화
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # 9채널을 3x3 subplot에
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        # i번째 채널의 스펙트럼 (freq, time)
        spec_ch = spec_9ch[i]
        
        # imshow를 이용한 시각화 (freq x time이므로 원래는 세로축이 freq, 가로축이 time)
        im = ax.imshow(spec_ch, aspect='auto', origin='lower', cmap='jet')
        
        # Grad-CAM heatmap 오버레이 (동일 크기)
        # alpha로 투명도 조절
        ax.imshow(gradcam_map, aspect='auto', origin='lower', cmap='magma', alpha=0.4)
        
        ax.set_title(f'Channel {i+1}')
        ax.axis('off')
    fig.suptitle('Grad-CAM Visualization on 9-Channel Spectrogram')
    plt.tight_layout()
    plt.show()

# -------------------- 메인 함수 --------------------
def main(edf_path, save_path=None):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

    # (예시) arousal_dir는 실제 상황에 맞게 수정
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

    # 모델 생성
    model = DeepSleepSota2D(in_channels=9).to(device)
    
    # 예시로 pretrained_path 지정 (사용 환경에 따라 수정)
    pretrained_path = "/home/honeynaps/data/saved_models_spec/ChunkSpecW2__f1_0.8007__PAD_tight_lr0.0010_fs50_ep17_auprc0.8471_th0.2412.pt"
    th = float(pretrained_path.split('_')[-1].replace('.pt', '').replace('th', ''))
    
    # 모델 로드
    state_dict = torch.load(pretrained_path, map_location=device)
    # state_dict가 'weights_only=True'로 저장되었는지 여부에 맞춰서 키를 맞춰야 함
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)

    # 추론(샘플 평가)
    y_preds, y_targets, y_probs, acc, precision, recall, fl = eval_fn2(model, val_loader, device, th=th)
    print(f"[Before Postprocess]\nAccuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fl:.4f}")

    # 후처리 및 이벤트 레벨 분석 (단일 파일이므로 y_preds[0], y_targets[0], y_probs[0]만 존재한다고 가정)
    y_pred = y_preds[0]
    y_target = y_targets[0]
    y_prob = y_probs[0]

    # 예: 길이 3.8초 미만의 이벤트는 제거
    y_pred = postprocess_arousal_preds(y_pred, min_len=3.8, fs=50)

    print("[After Postprocess]")
    acc2 = accuracy_score(y_target, y_pred)
    precision2 = precision_score(y_target, y_pred, zero_division=0)
    recall2 = recall_score(y_target, y_pred, zero_division=0)
    f12 = f1_score(y_target, y_pred, zero_division=0)
    print(f"Accuracy: {acc2:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}, F1: {f12:.4f}")

    # 이벤트 레벨 분석
    stats = event_level_analysis(y_pred, y_target, y_prob, 
                                xlsx_path=None,  # 혹은 excel 파일 경로
                                overlap_th=0.1,
                                return_stats=True)
    print("Event-level stats:", stats)

    # XML 저장 (옵션)
    if save_path is not None:
        xml_file = os.path.join(save_path, edf_name.replace(".edf", "_AROUS_PRED.xml"))
        save_to_xml(edf_path, y_pred, xml_file)
        print(f"Saved XML at: {xml_file}")

    # ---------------- Grad-CAM 시각화 ----------------
    # Grad-CAM을 볼 타겟 레이어를 지정 (UNet 기반이라면 마지막 conv 블록 등)
    # 예: model.decoder[-1].conv_block[-1] 처럼 구체적으로 잡아야 할 수도 있음
    # 아래는 예시로 model.decoder[-1] 사용:
    target_layer = model.decoder[-1]
    
    # val_loader 재사용해서 Grad-CAM 시각화
    visualize_gradcam(model, val_loader, device, target_layer)

    return acc2, precision2, recall2, f12, stats

if __name__ == "__main__":
    edf_path = "/home/honeynaps/data/GOLDEN/EDF2/SCH_F_60_NW_230921R4_NO.edf"
    save_path = '/home/honeynaps/data/shared/arousal'
    main(edf_path, save_path)
