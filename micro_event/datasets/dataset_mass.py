# dataset.py
import os
import numpy as np
import pyedflib
import torch
import sys

sys.path.append('/home/honeynaps/data/eis/SEED_pytorch')
from util import utils

class SleepEventDataset(torch.utils.data.Dataset):
    def _read_eeg_signal(self, signal_path):
        with pyedflib.EdfReader(signal_path) as file:
            channel_names = file.getSignalLabels()
            channel_to_extract = channel_names.index("EEG C3-CLE") 
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
            # Check
            # print(f"Channel extracted: {file.getLabel(channel_to_extract)}")

        # Particular fix for mass dataset:
        fs_old_round = int(np.round(fs_old))
        # Transform the original fs frequency with decimals to rounded version
        signal = utils.resample_signal_linear(
            signal, fs_old=fs_old, fs_new=fs_old_round
        )

        # Broand bandpass filter to signal
        signal = utils.broad_filter(signal, fs_old_round)

        # Now resample to the required frequency
        if self.fs != fs_old_round:
            # print(f"Resampling from {fs_old_round} Hz to required {self.fs} Hz")
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
        # print(f"N2 pages: {n2_pages.shape[0]}")
        # print(f"Whole-night pages: {all_pages.shape[0]}")
        # print(f"Hypnogram pages: {hypnogram.shape[0]}")
        # print(f"Marks {self.event_type.upper()} from E1: {marks_1.shape[0]}")

        # Save data
        ind_dict = {
            "signal": signal,
            "n2_pages": n2_pages,
            "all_pages": all_pages,
            "marks": marks_1,
            "hypnogram": hypnogram,
        }

        return ind_dict

    def __init__(self, data_dir, subject_ids, augmented_page=False,
                 border_size=2.6, normalize_clip=True, pages_subset="N2",
                 normalization_mode="N2", event_type='spindle', 
                 annotator='E1',
                 norm_mad=False,
                 page_duration=20, target_fs=200):
        self.data_dir            = data_dir
        self.subject_ids         = subject_ids
        self.event_type          = event_type.lower()
        self.annotator           = annotator.upper()
        self.fs                  = target_fs      
        self.subject_data        = {}
        self.page_subset         = pages_subset  # 'N2' or 'all'
        self.normalization_mode  = normalization_mode
        self.normalize_clip      = normalize_clip
        self.page_duration       = page_duration  # in seconds
        self.page_size           = int(self.fs * self.page_duration)  # in samples
        self.unknown_id          = "?"
        self.n2_id               = "2"  # N2 stage ID
        self.min_kc_duration     = 0.2
        self.aligned_downsample  = True 
        self.augmented_page      = augmented_page
        self.border_size         = int(np.round(border_size * self.fs))  # Convert to samples
        self.stride              = 8 
        self.norm_mad            = norm_mad

        for sid in self.subject_ids:
            # Construct file paths
            code = f"01-02-00{sid:02d}"
            psg_path = os.path.join(data_dir, f"{code} PSG.edf")
            state_path = os.path.join(data_dir, f"{code} Base.edf")
            if self.event_type == 'spindle':
                ann_path = os.path.join(data_dir, f"{code} Spindles_{self.annotator}.edf")
            elif self.event_type == 'kcomplex':
                ann_path = os.path.join(data_dir, f"{code} KComplexes_E1.edf")
            else:
                raise ValueError("event_type must be 'spindle' or 'kcomplex'")
            
            data = self._load_subject_data(psg_path, state_path, ann_path)
            self.subject_data[sid] = data

        self.global_std = self._calculate_global_std()

        print(f"Global std: {self.global_std:.2f} uV")
    
        self.signals, self.marks, self.page_masks = self._prepare_data()
        seg_has_event = self.marks.sum(axis=1) > 0
        self.pos_segments = int(seg_has_event.sum())
        self.neg_segments = int(self.marks.shape[0] - self.pos_segments)
        print(self.pos_segments, self.neg_segments)

    def __len__(self):
        return self.signals.shape[0]
    
    def __getitem__(self, idx):
        feat  = self.signals[idx]   # shape (total_length,)
        label = self.marks[idx]   # shape (total_length,)
        mask  = self.page_masks[idx]  # shape (total_length,)

        # 1. **Randomly crop** to a subsegment of length (page_size + 2*border_size)
        total_length = feat.shape[-1]
        crop_length = self.page_size + 2 * self.border_size
        if total_length > crop_length:
            # Choose a random start index for cropping
            max_offset = total_length - crop_length
            start = np.random.randint(0, max_offset + 1)
            end = start + crop_length
            feat = feat[start:end]
            label = label[start:end]
            mask = mask[start:end]

        # 2. **Remove border** regions from label and mask to get the central page region
        center_label = label[self.border_size : -self.border_size]   # shape = page_size
        center_mask  = mask[self.border_size : -self.border_size]    # shape = page_size

        # 3. **Downsample** the label and mask by the stride factor to match model output rate
        if self.aligned_downsample:
            # Aligned downsampling: use mean of each stride block then round to nearest integer
            # (Assumes page_size is an integer multiple of stride for perfect alignment)
            block_size = self.stride
            # If page_size isn't exactly divisible by stride, we handle the remainder by slicing
            trimmed_length = (len(center_label) // block_size) * block_size
            label_blocks = center_label[:trimmed_length].reshape(-1, block_size)
            mask_blocks  = center_mask[:trimmed_length].reshape(-1, block_size)
            # Compute mean in each block
            label_down = label_blocks.mean(axis=1)
            mask_down  = mask_blocks.mean(axis=1)
            # Round the label mean to 0 or 1 (for binary events)
            label_down = np.rint(label_down).astype(np.float32)
            # For the mask, round the mean (mask is mostly 0/1, so this effectively checks if the block is valid)
            mask_down = np.rint(mask_down).astype(np.float32)
            # If there were any leftover samples (non-divisible by stride), you could ignore or handle them as needed.
        else:
            # Non-aligned downsampling: take every `stride`-th sample (sub-sampling)
            label_down = center_label[:: self.stride].astype(np.float32)
            mask_down  = center_mask[:: self.stride].astype(np.float32)

        # 4. **Convert to PyTorch tensors** with appropriate shape and type
        feat_tensor = torch.from_numpy(feat).float()
        if feat_tensor.dim() == 1:
            feat_tensor = feat_tensor.unsqueeze(0)  # shape (1, length) for a single-channel signal
        # Label and mask tensors (for loss computation)
        label_tensor = torch.from_numpy(label_down).float()  # shape (downsampled_length,)
        mask_tensor  = torch.from_numpy(mask_down).float()   # shape (downsampled_length,)
        return feat_tensor, label_tensor, mask_tensor
        

    def _calculate_global_std(self):
        total_samples = 0
        sum_x = 0.0
        sum_x2 = 0.0
        for subject_id in self.subject_ids:
            ind_dict = self.subject_data[subject_id]
            x = ind_dict["signal"]

            if self.norm_mad:
                median_val = np.median(x)
                mad = np.median(np.abs(x - median_val))
                x = (x - median_val) / (1.4826 * mad)
                x *= 10
                ind_dict["signal"] = x

            # Only sleep
            hypno = ind_dict["hypnogram"]
            pages = np.concatenate(
                [np.where(hypno == lbl)[0] for lbl in ["1", "2", "3", "4", "R"]]
            )
            hypnogram_page_size = int(np.round(self.page_duration * self.fs))
            x = utils.extract_pages(x, pages, hypnogram_page_size).flatten()

            outlier_thr = np.percentile(np.abs(x), 99)
            print(f"Outlier threshold for subject {subject_id}: {outlier_thr:.2f} uV")
            x = x[np.abs(x) <= outlier_thr]
            total_samples += x.shape[0]
            sum_x += np.sum(x)
            sum_x2 += np.sum(x**2)
        mean_squared_x = sum_x2 / total_samples
        mean_x = sum_x / total_samples
        global_variance = mean_squared_x - (mean_x**2)
        global_std = np.sqrt(global_variance)

        return global_std

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
                pages_subset=self.page_subset,
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

if __name__ == "__main__":
    # Example usage
    data_dir = "/home/honeynaps/data/MASS_FILES/SS2"
    subject_ids = [1, 2, 3, 4]  # Example subject IDs
    dataset = SleepEventDataset(data_dir, subject_ids, event_type='kcomplex', annotator='E1')
    
    # Prepare the data
    x, y, page_mask = dataset._prepare_data()
    
    print("Signals shape:", x.shape)
    print("States shape:", y.shape)
    print("Page mask shape:", page_mask.shape)
