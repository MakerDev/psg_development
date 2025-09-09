import torch
import numpy as np
import random
from datetime import datetime

from .models.cnn_encoders import *
from .modules.preprocessing import *
from .utils.transforms import *
from .utils.tools import *
from .prep_window_wise import epoching_from_time
from .utils.post_process import run_postprocess

from .ProgNoti import ProgNoti


class SleepFinal :

    class _DataSet(torch.utils.data.Dataset) :

        def __init__(self, x, num_channels, transforms=None, missing_ch=[]):
            super().__init__()

            self.num_channels = num_channels
            self.transform  = transforms
            self.data_list  = [x]
            self.label_list = [np.zeros(x.shape[0])]
            self.missing_ch = missing_ch

            self.data_list  = [torch.tensor(data  , dtype=torch.float32) for data   in self.data_list ]
            self.label_list = [torch.tensor(labels, dtype=torch.long   ) for labels in self.label_list]
            self.data_list  = torch.concat(self.data_list, dim=0).unsqueeze(1)
            self.label_list = torch.concat(self.label_list, dim=0)

            self.data_list, self.label_list = self._group_data(self.data_list, self.label_list, 1)
            self._permute_data()
        #--INIT


        def _fill_missing_channels(self, recording, missing_channels):
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

            return filled_recording
        
        def _group_data(self, data, labels, n):
            grouped_data = []
            grouped_labels = []
            for idx in range(0, len(data) - n + 1):
                grouped_data.append(data[idx:idx+n]) 
                grouped_labels.append(labels[idx+n-1])  # Label for the last item in the group
            
            grouped_data = torch.stack(grouped_data)
            grouped_labels = torch.tensor(grouped_labels, dtype=torch.long)
            
            return grouped_data, grouped_labels
        #--DEF

        def _permute_data(self):     
            self.data_list = self.data_list.reshape(-1, 1, self.data_list.size(3), self.data_list.size(4))
            self.data_list = self.data_list.permute(0, 3, 1, 2)
        #--DEF

        def __len__(self):
            return len(self.label_list)
        #--DEF
        
        def __getitem__(self, idx):
            data = self.data_list[idx]
            label = self.label_list[idx]

            if self.transform:
                original_shape = data.shape

                if len(self.missing_ch) > 0:
                    data = data.reshape(9 - len(self.missing_ch), -1)
                    data = self._fill_missing_channels(data, self.missing_ch)
                    original_shape = (9, 1, 1500)
                else:
                    data = data.reshape(self.num_channels, -1)

                data, label = self.transform(data, label)
                data = data.reshape(original_shape)
            #--IF

            return data, label
        #--DEF
    #--CLASS


    def __init__(self, 
                 sigs        :dict, 
                 base_time   :datetime, 
                 start_time  :datetime=None,
                 model       :str='resnet18',
                 gpu         :int=0,
                 seed        :int=5,
                 num_channels:int=9,
                 fs          :int=50,
                 nofill      :bool=True,
                 tag         :str='',
                 progress    :ProgNoti=None,
                 missing_ch  :str='raise'
            ):

        self.sigs       = sigs
        self.base_time  = base_time
        if start_time :
            self.start_time = start_time
        else :
            self.start_time = base_time

        self.model        = model
        self.gpu          = gpu
        self.seed         = seed
        self.num_channels = num_channels
        self.fs           = fs
        self.nofill       = nofill
        self.tag          = tag

        # 알고리즘 진척 단계 출력용
        self.progress     = progress
    #--DEF


    def __call__(self, pretrained_dir, n_missing=0):

        if self.progress : self.progress.stepForward()
        transforms = ["NormaliseOnly"]
        transforms = build_transforms(transforms, n_channels=self.num_channels)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        # 채널 순서 배치 : 여기서 누락 채널 체크 추가 필요
        SID_SEQs = [ 'F3_2', 'F4_1', 'C3_2', 'C4_1', 'O1_2', 'O2_1', 'LOC', 'ROC', 'CHIN' ]
        data = [ None for _ in range(len(SID_SEQs)) ]
        for sid, sig in self.sigs.items() :
            i = SID_SEQs.index(sid)
            data[i] = sig
        #--FOR

        missing_channels = [i for i in range(9) if data[i] is None]

        data = [d for d in data if d is not None]
        data = np.array(data)

        if len(missing_channels) > 0:
            data = prep_psg_signal_with_missing(data, transpose=True, fs=self.fs, missing_channels=missing_channels)
        else:
            data = prep_psg_signal(data, transpose=True, fs=self.fs)
        X = epoching_from_time(data, self.base_time, self.start_time, sfreq=self.fs)

        if n_missing > 0:
            pretrained_model = f"pretrained_miss{n_missing}.pt"
        else:
            pretrained_model = "pretrained_asam_ver2.pt"

        pretrained_path = f'{pretrained_dir}/{pretrained_model}'

        num_channels = self.num_channels
        dataset = SleepFinal._DataSet(X, num_channels=num_channels, transforms=transforms, missing_ch=missing_channels)

        # 원하는 모델 아키텍쳐로 초기화 후 pretrained 모델 불러오기.
        if self.model == 'resnet18':
            model = resnet18(num_channels=num_channels, pretrained=False)
        elif self.model == 'resnet50':
            model = resnet50(num_channels=num_channels, pretrained=False)
        elif self.model == 'regnet128':
            model = regnet128(num_channels=num_channels, pretrained=False)
        elif self.model == 'regnet16':
            model = regnet16(num_channels=num_channels, pretrained=False)
        elif self.model == 'swin':
            model = swin_transformer(num_channels=num_channels)
        elif self.model == 'convnext':
            model = conv_next(num_channels=num_channels)

        if self.progress : self.progress.stepForward()
        device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

        num_signals = 1500 * (self.fs // 50)
        y_true, y_pred = [], []
        all_probs = []

        model.eval()
        with torch.no_grad():
            if self.progress : self.progress.stepForward()
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
            #--FOR

            if self.progress : self.progress.stepForward()
            y_pred = run_postprocess(y_pred, 6)
        #--WITH

        return y_pred, all_probs
    #--CALL

#--CLASS
