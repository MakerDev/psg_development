import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.utils.data import Dataset, DataLoader
import os
import utils
# ===== 1. 데이터 전처리 개선 =====

class ImprovedSleepEventDataset(Dataset):
    def __init__(self, data_dir, subject_ids, augmented_page=False,
                 border_size=2.6, normalize_clip=True,
                 normalization_mode="N2", event_type='spindle', annotator='E1',
                 page_duration=20, target_fs=200):
        self.data_dir = data_dir
        self.subject_ids = subject_ids
        self.event_type = event_type.lower()
        self.annotator = annotator.upper()
        self.fs = target_fs
        self.subject_data = {}
        self.normalization_mode = normalization_mode
        self.normalize_clip = normalize_clip
        self.page_duration = page_duration
        self.page_size = int(self.fs * self.page_duration)
        self.unknown_id = "?"
        self.n2_id = "2"
        self.min_kc_duration = 0.2
        self.aligned_downsample = True
        self.augmented_page = augmented_page
        self.border_size = int(np.round(border_size * self.fs))
        self.stride = 8
        
        # 데이터 로드
        for sid in self.subject_ids:
            code = f"01-02-00{sid:02d}"
            psg_path = os.path.join(data_dir, f"{code} PSG.edf")
            state_path = os.path.join(data_dir, f"{code} Base.edf")
            if self.event_type == 'spindle':
                ann_path = os.path.join(data_dir, f"{code} Spindles_{self.annotator}.edf")
            else:
                ann_path = os.path.join(data_dir, f"{code} KComplexes_E1.edf")
            
            data = self._load_subject_data(psg_path, state_path, ann_path)
            self.subject_data[sid] = data
        
        # Global STD 계산 (개선점 1: unbiased=False)
        self.global_std = self._calculate_global_std_improved()
        self.signals, self.states, self.page_masks = self._prepare_data()

    def __len__(self):
        return len(self.signals)

    def _calculate_global_std_improved(self):
        """TensorFlow와 동일한 population STD 계산"""
        total_samples = 0
        sum_x = 0.0
        sum_x2 = 0.0
        
        for subject_id in self.subject_ids:
            ind_dict = self.subject_data[subject_id]
            x = ind_dict["signal"]
            
            # Sleep stages만 추출
            hypno = ind_dict["hypnogram"]
            pages = np.concatenate(
                [np.where(hypno == lbl)[0] for lbl in ["1", "2", "3", "4", "R"]]
            )
            hypnogram_page_size = int(np.round(self.page_duration * self.fs))
            x = self._extract_pages(x, pages, hypnogram_page_size).flatten()
            
            outlier_thr = np.percentile(np.abs(x), 99)
            x = x[np.abs(x) <= outlier_thr]
            total_samples += x.shape[0]
            sum_x += np.sum(x)
            sum_x2 += np.sum(x**2)
        
        mean_squared_x = sum_x2 / total_samples
        mean_x = sum_x / total_samples
        global_variance = mean_squared_x - (mean_x**2)
        global_std = np.sqrt(global_variance)  # Population STD (unbiased=False와 동일)
        
        return global_std

    def _read_eeg_signal(self, signal_path):
        with pyedflib.EdfReader(signal_path) as file:
            channel_names = file.getSignalLabels()
            channel_to_extract = channel_names.index("EEG C3-CLE") 
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
            # Check
            print(f"Channel extracted: {file.getLabel(channel_to_extract)}")

        # Particular fix for mass dataset:
        fs_old_round = int(np.round(fs_old))
        # Transform the original fs frequency with decimals to rounded version
        signal = utils.resample_signal_linear(
            signal, fs_old=fs_old, fs_new=fs_old_round
        )

        # Broand bandpass filter to signal
        # signal = utils.broad_filter(signal, fs_old_round)
        signal = self._broad_filter_improved(signal, fs_old_round, lowcut=0.1, highcut=35)

        # Now resample to the required frequency
        if self.fs != fs_old_round:
            print(f"Resampling from {fs_old_round} Hz to required {self.fs} Hz")
            signal = utils.resample_signal(signal, fs_old=fs_old_round, fs_new=self.fs)
        else:
            print(f"Signal already at required {self.fs} Hz")

        signal = signal.astype(np.float32)
        return signal
    
    def _read_states_raw(self, stage_path):
        with pyedflib.EdfReader(stage_path) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])  # In seconds
        durations = np.round(np.array(annotations[1]))  # In seconds
        stages_str = annotations[2]
        # keep only 20s durations
        valid_idx = durations == self.page_duration
        onsets = onsets[valid_idx]
        stages_str = stages_str[valid_idx]
        stages_char = np.asarray([single_annot[-1] for single_annot in stages_str])
        # Sort by onset
        sorted_locs = np.argsort(onsets)
        onsets = onsets[sorted_locs]
        stages_char = stages_char[sorted_locs]
        # The hypnogram could start at a sample different from 0
        start_time = onsets[0]
        onsets_relative = onsets - start_time
        onsets_pages = np.round(onsets_relative / self.page_duration).astype(np.int32)
        n_scored_pages = (
            1 + onsets_pages[-1]
        )  # might be greater than onsets_pages.size if some labels are missing
        start_sample = int(start_time * self.fs)
        hypnogram = (n_scored_pages + 1) * [
            self.unknown_id
        ]  # if missing, it will be "?", we add one final '?'
        for scored_pos, scored_label in zip(onsets_pages, stages_char):
            hypnogram[scored_pos] = scored_label
        hypnogram = np.asarray(hypnogram)
        return hypnogram, start_sample
    
    def _fix_signal_and_states(self, signal, hypnogram, start_sample):
        # Crop start of signal
        signal = signal[start_sample:]
        # Find the largest valid sample, common in both signal and hypnogram, with an integer number of pages
        n_samples_from_signal = int(self.page_size * (signal.size // self.page_size))
        n_samples_from_hypnogram = int(hypnogram.size * self.page_size)
        n_samples_valid = min(n_samples_from_signal, n_samples_from_hypnogram)
        n_pages_valid = int(n_samples_valid / self.page_size)
        # Fix signal and hypnogram according to this maximum sample
        signal = signal[:n_samples_valid]
        hypnogram = hypnogram[:n_pages_valid]
        end_sample = (
            start_sample + n_samples_valid
        )  # wrt original beginning of recording, useful for marks
        return signal, hypnogram, end_sample
    
    def _hypnogram_selections(self, hypnogram):
        total_pages = hypnogram.size
        n2_pages = np.where(hypnogram == self.n2_id)[0].astype(np.int16)
        # Drop first and last page of the whole registers if they where selected.
        last_page = total_pages - 1
        n2_pages = n2_pages[(n2_pages != 0) & (n2_pages != last_page)]
        all_pages = np.arange(1, total_pages - 1, dtype=np.int16)
        return all_pages, n2_pages

    def _fix_marks(self, marks, start_sample, end_sample):
        marks = marks - start_sample  # reference to new start
        end_sample = end_sample - start_sample
        marks = utils.filter_stamps(marks, 0, end_sample - 1)  # avoid runaway
        return marks

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        with pyedflib.EdfReader(path_marks_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        offsets = onsets + durations
        marks_time = np.stack((onsets, offsets), axis=1)  # time-stamps
        # Transforms to sample-stamps
        marks = np.round(marks_time * self.fs).astype(np.int32)
        # Fix durations that are outside standards
        marks = utils.filter_duration_stamps(
            marks, self.fs, self.min_kc_duration, None
        )
        return marks
    
    def _load_subject_data(self, signal_path, stage_path, anno_path):
        signal = self._read_eeg_signal(signal_path)
        hypnogram, start_sample = self._read_states_raw(stage_path)
        signal, hypnogram, end_sample = self._fix_signal_and_states(
            signal, hypnogram, start_sample
        )
        all_pages, n2_pages = self._hypnogram_selections(hypnogram)
        marks_1 = self._read_marks(anno_path)
        marks_1 = self._fix_marks(marks_1, start_sample, end_sample)
        print(f"N2 pages: {n2_pages.shape[0]}")
        print(f"Whole-night pages: {all_pages.shape[0]}")
        print(f"Hypnogram pages: {hypnogram.shape[0]}")
        print(f"Marks {self.event_type.upper()} from E1: {marks_1.shape[0]}")

        # Save data
        ind_dict = {
            "signal": signal,
            "n2_pages": n2_pages,
            "all_pages": all_pages,
            "marks": marks_1,
            "hypnogram": hypnogram,
        }

        return ind_dict


    def _broad_filter_improved(self, signal, fs, lowcut=0.1, highcut=35):
        """TensorFlow와 동일한 필터링 구현"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(3, (low, high), btype='band')
        # Zero-phase filtering으로 위상 왜곡 방지
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    def _extract_pages(self, sequence, pages_indices, page_size, border_size=0):
        """TensorFlow 'SAME' 패딩과 동일한 페이지 추출"""
        sequence = np.asarray(sequence)
        pages_indices = np.asarray(pages_indices)
        
        # 개선점 2: TensorFlow SAME 패딩 에뮬레이션
        total_border = border_size
        if self.augmented_page:
            total_border = self.page_size // 2 + border_size
        
        pages_list = []
        for page in pages_indices:
            sample_start = page * page_size - total_border
            sample_end = (page + 1) * page_size + total_border
            
            # TensorFlow style padding
            if sample_start < 0:
                # 앞쪽 패딩
                pad_front = -sample_start
                page_signal = sequence[:sample_end]
                page_signal = np.pad(page_signal, (pad_front, 0), mode='constant')
            elif sample_end > len(sequence):
                # 뒤쪽 패딩
                pad_back = sample_end - len(sequence)
                page_signal = sequence[sample_start:]
                page_signal = np.pad(page_signal, (0, pad_back), mode='constant')
            else:
                page_signal = sequence[sample_start:sample_end]
            
            pages_list.append(page_signal)
        
        pages_data = np.stack(pages_list, axis=0)
        return pages_data
    
    def __getitem__(self, idx):
        feat = self.signals[idx]
        label = self.states[idx]
        mask = self.page_masks[idx]
        
        # Random crop
        total_length = feat.shape[-1]
        crop_length = self.page_size + 2 * self.border_size
        if total_length > crop_length:
            max_offset = total_length - crop_length
            start = np.random.randint(0, max_offset + 1)
            end = start + crop_length
            feat = feat[start:end]
            label = label[start:end]
            mask = mask[start:end]
        
        # Remove border regions
        center_label = label[self.border_size : -self.border_size]
        center_mask = mask[self.border_size : -self.border_size]
        
        # Aligned downsampling (TensorFlow와 동일)
        if self.aligned_downsample:
            block_size = self.stride
            trimmed_length = (len(center_label) // block_size) * block_size
            label_blocks = center_label[:trimmed_length].reshape(-1, block_size)
            mask_blocks = center_mask[:trimmed_length].reshape(-1, block_size)
            
            label_down = np.rint(label_blocks.mean(axis=1)).astype(np.float32)
            mask_down = np.rint(mask_blocks.mean(axis=1)).astype(np.float32)
        else:
            label_down = center_label[::self.stride].astype(np.float32)
            mask_down = center_mask[::self.stride].astype(np.float32)
        
        # Convert to tensors
        feat_tensor = torch.from_numpy(feat).float()
        if feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(0)
        
        label_tensor = torch.from_numpy(label_down).float()
        mask_tensor = torch.from_numpy(mask_down).float()
        
        return feat_tensor, label_tensor, mask_tensor
    
    def _prepare_data(self):
        """
        Prepare the dataset by extracting signals and events for each subject.
        This method should be called after initializing the dataset.
        """
        signals, states, page_masks = [], [], []

        for sid, _ in self.subject_data.items():
            signal, stats, page_mask = self._get_subject_data(
                sugject_id=sid,
                augmented_page=self.augmented_page,
                border_size=2.6,
                forced_mark_separation_size=0,
                pages_subset=self.normalization_mode,
                normalize_clip=self.normalize_clip,
                normalization_mode=self.normalization_mode,
                return_page_mask=True
            )

            signals.append(signal)
            states.append(stats)
            page_masks.append(page_mask)

        x = np.concatenate(signals, axis=0)
        y = np.concatenate(states, axis=0)
        page_mask = np.concatenate(page_masks, axis=0)

        return x, y, page_mask
    
    def _get_subject_data(self,
                          sugject_id,
                          augmented_page=False,
                          border_size=2.6,
                          forced_mark_separation_size=0,
                          pages_subset="N2",
                          normalize_clip=True,
                          normalization_mode="N2",
                          return_page_mask=False):
        ind_dict = self.subject_data[sugject_id]
        signal   = ind_dict["signal"]
        marks    = ind_dict["marks"]
        pages    = ind_dict["n2_pages"] if pages_subset == "N2" else ind_dict["all_pages"]

        if forced_mark_separation_size > 0:
            marks = utils.stamp2seq_with_separation(marks, 0, signal.shape[0] - 1, forced_mark_separation_size)
        else:
            marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)        
            
        pages = pages.astype(np.int32)
        pages_start = pages * self.page_size
        pages_end = (pages + 1) * self.page_size - 1
        pages_stamps = np.stack([pages_start, pages_end], axis=1).astype(np.int32)
        page_mask = utils.stamp2seq(pages_stamps, 0, signal.shape[0] - 1)

        border_size = int(np.round(border_size * self.fs))  # Convert to samples
        # Compute border to be added
        if augmented_page:
            total_border = self.page_size // 2 + border_size
        else:
            total_border = border_size

        if normalize_clip:
            if normalization_mode != "N2":
                # Normalization with stats from pages containing true events.
                # Normalize using stats from pages with true events.
                tmp_pages = ind_dict["all_pages"]
                activity = utils.extract_pages(
                    marks, tmp_pages, self.page_size, border_size=0
                )
                activity = activity.sum(axis=1)
                activity = np.where(activity > 0)[0]
                tmp_pages = tmp_pages[activity]
                signal, _ = utils.norm_clip_signal(
                    signal,
                    tmp_pages,
                    self.page_size,
                    norm_computation='global',
                    global_std=self.global_std,
                    clip_value=10,
                )
            else:
                n2_pages = ind_dict["n2_pages"]
                signal, _ = utils.norm_clip_signal(
                    signal,
                    n2_pages,
                    self.page_size,
                    norm_computation='global',
                    global_std=self.global_std,
                    clip_value=10,
                )

        # Extract segments
        signal = utils.extract_pages(
            signal, pages, self.page_size, border_size=total_border
        )
        marks = utils.extract_pages(
            marks, pages, self.page_size, border_size=total_border
        )
        page_mask = utils.extract_pages(
            page_mask, pages, self.page_size, border_size=total_border
        )

        # Set dtype
        signal = signal.astype(np.float32)
        marks = marks.astype(np.int8)
        page_mask = page_mask.astype(np.int8)

        if return_page_mask:
            return signal, marks, page_mask
        else:
            return signal, marks


# ===== 2. 모델 아키텍처 개선 =====

class ImprovedSleepEventDetector(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        
        # 개선점 5: Glorot/Xavier 초기화 사용
        def init_weights(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Local Encoding CNN Stage
        self.input_bn = nn.BatchNorm1d(input_channels)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(64)
        
        # Multi-dilation conv blocks
        self.conv_mdb1 = self._make_conv_md_block(64, 128)
        self.conv_mdb2 = self._make_conv_md_block(128, 256)
        
        # BLSTM Contextualization Stage (개선된 LSTM 설정)
        self.blstm1 = nn.LSTM(input_size=256, hidden_size=256, 
                              bidirectional=True, batch_first=True,
                              dropout=0.0)  # 첫 번째 레이어는 dropout 없음
        self.blstm2 = nn.LSTM(input_size=512, hidden_size=256, 
                              bidirectional=True, batch_first=True,
                              dropout=0.0)  # 두 번째 레이어도 dropout 없음
        
        self.dropout_blstm1 = nn.Dropout(p=0.2)
        self.dropout_blstm2 = nn.Dropout(p=0.5)
        
        # 1x1 conv to reduce features
        self.lin_proj = nn.Conv1d(512, 128, kernel_size=1)
        self.lin_proj_bn = nn.BatchNorm1d(128)
        self.dropout_proj = nn.Dropout(p=0.5)
        
        # Classification stage with proper initialization
        self.classifier = nn.Conv1d(128, 2, kernel_size=1)
        
        # 개선점 5: positive class 초기화
        init_positive_proba = 0.5
        bias_init = -np.log((1 - init_positive_proba) / init_positive_proba)
        with torch.no_grad():
            self.classifier.bias[1] = bias_init
        
        # Xavier 초기화 적용
        self.apply(init_weights)
    
    def _make_conv_md_block(self, in_channels, out_channels):
        """Multi-Dilated Convolutional Block"""
        assert out_channels % 4 == 0
        branch_out = out_channels // 4
        dilations = [1, 2, 4, 8]
        
        branches = nn.ModuleList()
        for d in dilations:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_out, kernel_size=3, 
                         dilation=d, padding=d),
                nn.BatchNorm1d(branch_out),
                nn.ReLU(inplace=True),
                nn.Conv1d(branch_out, branch_out, kernel_size=3, 
                         dilation=d, padding=d),
                nn.BatchNorm1d(branch_out),
                nn.ReLU(inplace=True)
            )
            branches.append(branch)
        
        fuse = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        return nn.ModuleDict({'branches': branches, 'fuse': fuse})
    
    def forward(self, x):
        # Input shape: [batch, channels, time]
        x = self.input_bn(x)
        
        # CNN stage
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.avg_pool1d(x, kernel_size=2)
        
        # First MDB + pool
        branch_outs = [branch(x) for branch in self.conv_mdb1['branches']]
        x = torch.cat(branch_outs, dim=1)
        x = self.conv_mdb1['fuse'](x)
        x = F.avg_pool1d(x, kernel_size=2)
        
        # Second MDB + pool
        branch_outs = [branch(x) for branch in self.conv_mdb2['branches']]
        x = torch.cat(branch_outs, dim=1)
        x = self.conv_mdb2['fuse'](x)
        x = F.avg_pool1d(x, kernel_size=2)
        
        # Transpose for LSTM: [batch, time, features]
        x = x.transpose(1, 2)
        
        # BLSTM layers with proper dropout
        x, _ = self.blstm1(x)
        x = self.dropout_blstm1(x)
        x, _ = self.blstm2(x)
        x = self.dropout_blstm2(x)
        
        # Project back to conv format
        x = x.transpose(1, 2)
        x = F.relu(self.lin_proj_bn(self.lin_proj(x)))
        x = self.dropout_proj(x)
        
        # Classification
        logits = self.classifier(x)
        return logits.transpose(1, 2)  # [batch, time, 2]

# ===== 3. 손실 함수 개선 =====

class ImprovedMaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets, mask):
        """
        개선점 6: TensorFlow와 동일한 Masked Focal Loss 구현
        """
        B, T, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1).long()
        mask_flat = mask.reshape(-1)
        
        # Softmax와 log_softmax 계산
        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = torch.exp(log_probs)
        
        # One-hot encoding
        targets_onehot = F.one_hot(targets_flat, num_classes=C).float()
        
        # Focal loss 계산
        pt = (probs * targets_onehot).sum(dim=-1)
        alpha_t = self.alpha * targets_onehot[:, 1] + (1 - self.alpha) * targets_onehot[:, 0]
        focal_factor = alpha_t * (1 - pt) ** self.gamma
        
        # Cross-entropy
        ce = -(targets_onehot * log_probs).sum(dim=-1)
        loss = focal_factor * ce
        
        # Mask 적용 (TensorFlow와 동일한 방식)
        loss = loss * mask_flat
        
        # 평균 계산 (mask가 적용된 영역에 대해서만)
        return loss.sum() / mask_flat.sum()

# ===== 4. 학습 설정 개선 =====

def train_improved_model(model, train_loader, val_loader, device, epochs=100):
    """
    개선점 3: BN 안정화를 위한 실제 배치 크기 사용
    개선점 4: 검증 세트에서 최적 임계값 탐색
    """
    criterion = ImprovedMaskedFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 학습률 스케줄러 (TensorFlow와 유사한 step-based)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_f1 = 0
    best_threshold = 0.5
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        
        for X, y, mask in train_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            logits = model(X)
            
            if logits.ndim > 2:
                logits = logits.squeeze(1)
            
            # Border cropping
            logits = logits[:, 65:-65]
            if y.ndim > 2:
                y = y.squeeze(1)
                mask = mask.squeeze(1)
            
            loss = criterion(logits, y, mask)
            loss.backward()
            
            # Gradient clipping (TensorFlow와 유사)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item() * X.size(0)
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        # Validation with threshold optimization
        model.eval()
        all_probs = []
        all_labels = []
        all_masks = []
        
        with torch.no_grad():
            for X, y, mask in val_loader:
                X, y, mask = X.to(device), y.to(device), mask.to(device)
                
                logits = model(X)
                if logits.ndim > 2:
                    logits = logits.squeeze(1)
                logits = logits[:, 65:-65]
                
                if mask.ndim > 2:
                    mask = mask.squeeze(1)
                if y.ndim > 2:
                    y = y.squeeze(1)
                
                probs = torch.softmax(logits, dim=-1)[..., 1]
                
                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())
                all_masks.append(mask.cpu())
        
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        all_masks = torch.cat(all_masks)
        
        # 개선점 4: 최적 임계값 탐색
        valid_mask = all_masks.bool()
        probs_valid = all_probs[valid_mask].numpy()
        labels_valid = all_labels[valid_mask].numpy()
        
        # PR curve를 통한 최적 임계값 탐색
        precisions, recalls, thresholds = precision_recall_curve(labels_valid, probs_valid)
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = f1s.argmax()
        best_threshold = thresholds[best_idx]
        best_f1 = f1s[best_idx]
        
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_threshold': best_threshold,
                'best_f1': best_f1,
                'epoch': epoch
            }, 'best_model.pth')
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Val F1 = {best_f1:.4f} (threshold = {best_threshold:.3f})")
    
    return best_threshold

# ===== 5. 후처리 개선 =====

class ImprovedPostProcessor:
    def __init__(self, fs=200, stride=8):
        self.fs = fs
        self.stride = stride
        self.fs_output = fs
        self.fs_input = fs // stride
    
    def proba2stamps(self, proba_data, threshold, min_duration=0.3, max_duration=3.0):
        """
        개선된 후처리: TensorFlow와 동일한 이벤트 스탬프 변환
        """
        # Low threshold for duration
        low_thr = threshold * 0.85
        
        # Binarization
        proba_bin_high = (proba_data >= threshold).astype(np.int32)
        proba_bin_low = (proba_data >= low_thr).astype(np.int32)
        
        # Convert to stamps
        stamps_low = self._seq2stamp(proba_bin_low)
        stamps_high = self._seq2stamp(proba_bin_high)
        
        if len(stamps_low) == 0 or len(stamps_high) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        
        # Keep only stamps that surpass high threshold
        valid_stamps = []
        for stamp_low in stamps_low:
            # Check if this low stamp contains any high stamp
            overlap = False
            for stamp_high in stamps_high:
                if stamp_high[0] >= stamp_low[0] and stamp_high[1] <= stamp_low[1]:
                    overlap = True
                    break
            if overlap:
                valid_stamps.append(stamp_low)
        
        if len(valid_stamps) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        
        stamps = np.array(valid_stamps)
        
        # Duration filtering
        stamps = self._filter_duration_stamps(
            stamps, self.fs_input, min_duration, max_duration
        )
        
        # Upsampling to original frequency
        stamps = stamps * self.stride
        stamps[:, 1] = stamps[:, 1] + self.stride - 1
        
        return stamps.astype(np.int32)
    
    def _seq2stamp(self, sequence):
        """Binary sequence to stamps conversion"""
        if len(sequence) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        
        diff = np.diff(np.concatenate([[0], sequence, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        if len(starts) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        
        return np.stack([starts, ends], axis=1)
    
    def _filter_duration_stamps(self, stamps, fs, min_duration, max_duration):
        """Filter stamps by duration"""
        if len(stamps) == 0:
            return stamps
        
        durations = (stamps[:, 1] - stamps[:, 0] + 1) / fs
        
        # Remove too short
        valid_idx = durations >= min_duration
        stamps = stamps[valid_idx]
        durations = durations[valid_idx]
        
        # Handle too long
        if max_duration is not None:
            # Remove extremely long (>2x max)
            valid_idx = durations <= 2 * max_duration
            stamps = stamps[valid_idx]
            durations = durations[valid_idx]
            
            # Crop long events to max_duration
            excess = durations - max_duration
            excess = np.clip(excess, 0, None)
            half_remove = ((fs * excess + 1) / 2).astype(np.int32)
            stamps[:, 0] = stamps[:, 0] + half_remove
            stamps[:, 1] = stamps[:, 1] - half_remove
        
        return stamps

# ===== 사용 예시 =====

if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    event_type = 'kcomplex'

    # 데이터셋 생성
    train_dataset = ImprovedSleepEventDataset(
        data_dir="/home/honeynaps/data/MASS_FILES/SS2",
        subject_ids=list(range(1, 16)),
        augmented_page=True,
        event_type=event_type
    )
    
    val_dataset = ImprovedSleepEventDataset(
        data_dir="/home/honeynaps/data/MASS_FILES/SS2",
        subject_ids=list(range(16, 20)),
        augmented_page=False,
        normalization_mode="N2",
        event_type=event_type
    )
    
    # 데이터로더 생성 (개선점 3: 실제 배치 크기 사용)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # gradient accumulation 대신 실제 배치 크기 사용
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 생성 및 학습
    model = ImprovedSleepEventDetector(input_channels=1).to(device)
    
    # 학습 (최적 임계값 반환)
    best_threshold = train_improved_model(
        model, train_loader, val_loader, device, epochs=100
    )
    
    print(f"Best threshold found: {best_threshold:.3f}")
    
    # 후처리기 생성
    postprocessor = ImprovedPostProcessor()
    
    # 추론 예시
    model.eval()
    with torch.no_grad():
        for X, _, _ in val_loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=-1)[..., 1]
            
            # 각 배치 아이템에 대해 후처리
            for i in range(probs.shape[0]):
                prob_seq = probs[i].cpu().numpy()
                stamps = postprocessor.proba2stamps(
                    prob_seq, 
                    threshold=best_threshold,
                    min_duration=0.3,
                    max_duration=3.0
                )
                print(f"Detected {len(stamps)} events")
            break