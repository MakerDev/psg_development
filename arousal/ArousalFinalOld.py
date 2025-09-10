import torch
import numpy as np
import os
from datetime import datetime


from utils.transforms import build_transforms
from utils.tools import pad_signals
from models.DeepSleepSota import DeepSleepNetSota
from models.DeepSleepSota2D import DeepSleepSota2D
from prep_arousal_ver3 import moving_window_mean_rms_norm
from prep_spectrogram import make_spectrogram
from common.seed import set_seed
from common.prog_noti import ProgNoti
from common.eval_utils import find_events, combine_two_models_events


class ArousalFinal :
    class _DataSet(torch.utils.data.Dataset):
        def __init__(self, name, x, y, transforms = None):
            super().__init__()

            self.name = name
            self.x, self.y = x, y
            self.transforms = transforms
        #--DEF

        def __len__(self):
            return 1
        #--DEF 
        
        
        def __getitem__(self, idx):
            if self.transforms is not None:
                x, y = self.transforms(self.x, self.y)
            else:
                x, y = self.x, self.y
            return self.name, x, y
        #--DEF
    #--CLASS


    def __init__(self, 
                 sigs        :dict, 
                 base_time   :datetime, 
                 start_time  :datetime=None,
                 min_secs    :int=3,
                 gpu         :int=0,
                 seed        :int=5,
                 num_channels:int=9,
                 fs          :int=50,
                 ver         :int=3,
                 type        :str='time',
                 tag         :str='',
                 progress    :ProgNoti=None
            ):

        self.sigs       = sigs
        self.base_time  = base_time
        if start_time :
            self.start_time = start_time
        else :
            self.start_time = base_time
        self.min_secs     = min_secs

        self.type         = type
        self.ver          = ver
        self.gpu          = gpu
        self.seed         = seed
        self.num_channels = num_channels
        self.fs           = fs
        self.tag          = tag
        self.is_ensemble  = self.type in ['union', 'intersection']

        # 알고리즘 진척 단계 출력용
        self.progress     = progress

        if self.type not in ['time', 'spec', 'union', 'intersection']:
            raise ValueError(f"Invalid type: {self.type}")
    #--DEF


    def map_spec_pred_to_time(
        self,
        pred_1d,
        times,
        total_samples,  
        fs=50,          
        nperseg=50,     
        mode='average'
    ):
        half_win_sec = nperseg / (2.0 * fs) 
        
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
                y_time[start_idx:end_idx] = np.maximum(
                    y_time[start_idx:end_idx],
                    pred_1d[i]
                )

        if mode == 'average':
            nonzero_mask = (count > 0)
            y_time[nonzero_mask] /= count[nonzero_mask]

        return y_time

    def postprocess_arousal_preds(self, preds, min_len=5, fs=50):
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
                        if end_idx >= start_idx:
                            new_preds[start_idx: end_idx + 1] = 1
                    
                    in_event = False

        return new_preds


    def evaluate_time_model(self, model, x, label, threshold):
        logits = model(x, True)

        preds = (logits > threshold).cpu().numpy().astype(int)
        preds = preds.squeeze()
        label = label.squeeze()

        pad_mask = label != -1
        label = label[pad_mask]
        preds = preds[pad_mask]

        preds = self.postprocess_arousal_preds(preds, min_len=5, fs=self.fs)

        return preds

    def evaluate_spec_model(self, model, x, th, times, total_samples):
        y_pred_2d = model(x)  
        y_pred_1d = y_pred_2d.mean(dim=2)
        y_pred_1d = y_pred_1d.squeeze()  
        y_pred = y_pred_1d.cpu()
        y_pred = (y_pred > th).numpy().astype(int)
        y_pred = self.map_spec_pred_to_time(pred_1d=y_pred, times=times, 
                                            total_samples=total_samples, fs=50, nperseg=50)
        y_pred = (y_pred > 0.5).astype(int)
        preds = self.postprocess_arousal_preds(y_pred, min_len=3, fs=self.fs)

        return preds

    def fill_missing_channels(self, recording, missing_channels): 
        recording = torch.tensor(recording.T)
        missing_channels = torch.tensor(missing_channels)
        filled_recording = torch.zeros((9, recording.shape[1]), dtype=recording.dtype)

        eeg_indices = torch.tensor(list(range(6)))
        missing_eeg = eeg_indices[torch.isin(eeg_indices, missing_channels)]
        remain_eeg = eeg_indices[~torch.isin(eeg_indices, missing_channels)]
        
        if len(remain_eeg) >= 1:
            eeg_mean = torch.mean(recording[remain_eeg], dim=0, keepdim=True)
            filled_recording[missing_eeg] = eeg_mean

        eog_indices = torch.tensor([6, 7])
        missing_eog = eog_indices[torch.isin(eog_indices, missing_channels)]
        remain_eog = eog_indices[~torch.isin(eog_indices, missing_channels)]
        
        if len(missing_eog) >= 1 and len(remain_eog) >= 1:
            ref_eog = remain_eog[0]
            filled_recording[missing_eog] = recording.clone()[ref_eog]

        return filled_recording.T
    
    def __build_model__(self, missing_channels, pretrained_dir, device):
        if self.is_ensemble:
            model_time = DeepSleepNetSota(n_channels=9)
            model_spec = DeepSleepSota2D(in_channels=9)
            pretrained_path_time = f'{pretrained_dir}/deepsleep_tight_asam_0.6587.pt'
            pretrained_path_spec = f'{pretrained_dir}/deepsleep_spec_0.923.pt'

            if len(missing_channels) > 0:
                n_missing = len(missing_channels)
                pretrained_files = os.listdir(pretrained_dir)
                pretrained_file = [f for f in pretrained_files if f"miss{n_missing}" in f and 'time' in f][0]
                pretrained_path_time = f'{pretrained_dir}/{pretrained_file}'
                pretrained_file = [f for f in pretrained_files if f"miss{n_missing}" in f and 'spec' in f][0]
                pretrained_path_spec = f'{pretrained_dir}/{pretrained_file}'

            th_time = float(pretrained_path_time.split('_')[-1].replace('.pt', '').replace('th', ''))
            th_spec = float(pretrained_path_spec.split('_')[-1].replace('.pt', '').replace('th', ''))

            model_time = model_time.to(device)
            model_time.load_state_dict(torch.load(pretrained_path_time, map_location=device))
            model_time.eval()

            model_spec = model_spec.to(device)
            model_spec.load_state_dict(torch.load(pretrained_path_spec, map_location=device))
            model_spec.eval()

            return None, None, model_time, th_time, model_spec, th_spec
        
        # Not ensemble
        if self.type == "time" and len(missing_channels) == 0:
            model = DeepSleepNetSota(n_channels=9)
        elif self.type == "spec":
            model = DeepSleepSota2D(in_channels=9)

        if self.type == "time":
            pretrained_path = f'{pretrained_dir}/deepsleep_tight_asam_0.6587.pt'
        elif self.type == "spec":
            pretrained_path = f'{pretrained_dir}/deepsleep_spec_0.923.pt'

        if len(missing_channels) > 0:
            n_missing = len(missing_channels)
            pretrained_files = os.listdir(pretrained_dir)
            pretrained_file = [f for f in pretrained_files if f"miss{n_missing}" in f and self.type in f][0]
            pretrained_path = f'{pretrained_dir}/{pretrained_file}'

        threshold = float(pretrained_path.split('_')[-1].replace('.pt', '').replace('th', ''))

        model = model.to(device)
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model.eval()

        return model, threshold, None, None, None, None
    
    def __call__(self, pretrained_dir):
        if self.progress : self.progress.stepForward()
        transforms = ["NormaliseOnly"]
        transforms = build_transforms(transforms, n_channels=self.num_channels)
        prep_fn = moving_window_mean_rms_norm

        set_seed(self.seed)

        # 채널 순서 배치 : 여기서 누락 채널 체크 추가 필요
        SID_SEQs = ['F3_2', 'F4_1', 'C3_2', 'C4_1', 'O1_2', 'O2_1', 'LOC', 'ROC', 'CHIN' ]
        data = [ None for _ in range(len(SID_SEQs)) ]
        for sid, sig in self.sigs.items() :
            i = SID_SEQs.index(sid)
            data[i] = sig
        #--FOR
        missing_channels = [i for i in range(9) if data[i] is None]

        data = [d for d in data if d is not None]
        data = np.array(data)

        if self.progress : self.progress.stepForward()
        start_sec = (self.start_time - self.base_time).total_seconds()
        start_idx = int(start_sec * self.fs)

        x = prep_fn(data, fs=self.fs)
        x = x.T
        x = x[start_idx:]
        total_samples = len(x)
        
        if len(missing_channels) > 0:
            x = self.fill_missing_channels(x, missing_channels)

        if self.type == 'time':
            max_len = 2**22 if self.fs == 100 else 2**21
            x, y = pad_signals(x, np.zeros(x.shape[0]), max_len)
        elif self.type == 'spec':
            transforms = None
            x, y, times = make_spectrogram(x, fs=self.fs, 
                                           nperseg=50,
                                           noverlap=25)
        else:
            max_len = 2**22 if self.fs == 100 else 2**21
            x_spec, y, times = make_spectrogram(x, fs=self.fs, 
                                                nperseg=50,
                                                noverlap=25)
            x, y_time = pad_signals(x, np.zeros(x.shape[0]), max_len)
            
        if self.progress : self.progress.stepForward()

        if not self.is_ensemble:
            dataset = ArousalFinal._DataSet('SOMNUM', x, y, transforms=transforms)
        else:
            # y에는 아무처리도 하지 않으므로, x_spec을 치팅으로 전달
            dataset = ArousalFinal._DataSet('SOMNUM', x, x_spec, transforms=transforms)
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')

        model, threshold, model_time, th_time, model_spec, th_spec = self.__build_model__(missing_channels, pretrained_dir, device)

        start_points, end_points = None, None
        with torch.no_grad():
            for _, data, label in loader :
                if self.progress : self.progress.stepForward()
                data = data.to(device) # batch size 1
                label = label.to(device)

                if self.type == "time":
                    preds = self.evaluate_time_model(model, data, label, threshold)
                elif self.type == "spec":
                    preds = self.evaluate_spec_model(model, data, threshold, times, total_samples)
                else:
                    preds_time = self.evaluate_time_model(model_time, data, y_time, th_time)
                    preds_spec = self.evaluate_spec_model(model_spec, label, th_spec, times, total_samples)
                    time_events = find_events(preds_time)
                    spec_events = find_events(preds_spec)

                    if len(time_events) == 0 and len(spec_events) == 0:
                        raise ValueError("No events found")
                    elif len(time_events) > len(spec_events) and len(spec_events) < 20:
                        spec_events = time_events
                    elif len(spec_events) > len(time_events) and len(time_events) < 20:
                        time_events = spec_events

                    preds, _ = combine_two_models_events(spec_events, time_events, total_samples, mode=self.type)

                print(len(preds))
                diff_y = np.diff(np.concatenate([[0], preds, [0]]))  
                start_points = np.where(diff_y ==  1)[0]
                end_points   = np.where(diff_y == -1)[0]
            #--FOR
        #--WITH

        if self.progress : self.progress.stepForward()
        base_sec = start_sec
        arous = []
        if isinstance(start_points, np.ndarray) and isinstance(end_points, np.ndarray) :
            for start_idx, end_idx in zip(start_points, end_points):
                start_sec = start_idx / self.fs
                end_sec = end_idx / self.fs
                duration = end_sec - start_sec

                if duration < self.min_secs:
                    continue

                arous.append((start_sec + base_sec, duration))
            #--FOR
        #--WITH

        return arous
    #--CALL

#--CLASS
