from .config import *

import mne 
from mne.io import read_raw_edf
from os.path import basename
import numpy as np

def load(path, preload=False, resample=False, preset=False, exclude=False, missing_ch='raise'):
    def virtual_key_mapping(ch_names, preset):
        mapping = dict()
        for v_key in PRESET[preset]:
            candidate = list(set(ch_names) & set(VIRTUAL_CH_KEY[v_key]))
            if len(candidate) >= 1: 
                mapping[candidate.pop()] = v_key
        return mapping

    def replace_key_mapping(target_key, edf_keys):
        replace_key_map = dict()
        for key in target_key:
            for r_key in REPLACE_KEY[key]:
                if r_key in edf_keys and key not in replace_key_map.keys():
                    replace_key_map[key] = r_key
        return replace_key_map

    try:
        mne.utils.set_config('MNE_USE_CUDA', 'true')  
        mne.cuda.init_cuda(verbose=False)
        raw = read_raw_edf(path, verbose=False, preload=preload)
    except:
        raw = read_raw_edf(path, verbose=False, preload=preload)

    if preset:
        mapping = virtual_key_mapping(raw.ch_names, preset)
        raw.rename_channels(mapping)

    n_missing_ch = 0

    try:
        if exclude and preset:
            raw.pick(PRESET[preset])

        target_key = list(set(PRESET[preset]) - set(raw.ch_names))
        replace_map = replace_key_mapping(target_key, raw.ch_names)

        ch_names = [key for key in replace_map] + raw.ch_names
        ch_picks = [raw.ch_names.index(ch) for ch in replace_map.values()] + \
                [raw.ch_names.index(ch) for ch in raw.ch_names]
        ch_types = ['mag' for _ in replace_map] + ['mag' for _ in raw.ch_names]

        if set(PRESET[preset]) != set(ch_names):
            raise RuntimeError(
                '{0} file doesn\'t satisfied with PRESET {1} PRESET KEYS: {2} EDF KEYS   : {3}'.format(
                    basename(path), preset, PRESET[preset], ch_names
                )
            )
        info = mne.create_info(ch_names=ch_names, sfreq=raw.info['sfreq'], ch_types=ch_types)
        info.set_meas_date(raw.info['meas_date'])
        data = raw.get_data(picks=ch_picks)
        raw = mne.io.RawArray(data, info, verbose=False)
    except Exception as e:
        if missing_ch == 'raise' and exclude:
            if set(PRESET[preset]) != set(raw.ch_names):
                raise RuntimeError(
                    '{0} file doesn\'t satisfied with PRESET {1} PRESET KEYS: {2} EDF KEYS   : {3}'.format(
                        basename(path), preset, PRESET[preset], raw.ch_names
                    )
                )
        # missing_ch 가 0/1/2 (Zero/Method1/Method2) 로 지정된 경우
        elif isinstance(missing_ch, int) and missing_ch in [0, 1, 2] and exclude:
            needed_channels = PRESET[preset]
            existing = set(raw.ch_names)
            needed  = set(needed_channels)
            missing = list(needed - existing)

            n_missing_ch = len(missing)

            if len(missing) > 0:
                old_data = raw.get_data() 
                old_ch_names = list(raw.ch_names)
                n_times = old_data.shape[1]
                sfreq = raw.info['sfreq']

                full_data = np.zeros((len(needed_channels), n_times), dtype=old_data.dtype)

                ch_index_map = { ch_name: i for i, ch_name in enumerate(needed_channels) }
                for i, ch_name in enumerate(old_ch_names):
                    new_idx = ch_index_map[ch_name]
                    full_data[new_idx, :] = old_data[i, :]

                missing_indices = [ch_index_map[ch] for ch in missing]
                remain_indices  = [ch_index_map[ch] for ch in needed_channels if ch not in missing]

                eeg_range = range(0, 6)
                eog_range = range(6, 8)
                emg_range = [8]        

                missing_eeg = [mi for mi in missing_indices if mi in eeg_range]
                missing_eog = [mi for mi in missing_indices if mi in eog_range]
                missing_emg = [mi for mi in missing_indices if mi in emg_range]

                remain_eeg = [ri for ri in remain_indices if ri in eeg_range]
                remain_eog = [ri for ri in remain_indices if ri in eog_range]
                remain_emg = [ri for ri in remain_indices if ri in emg_range]

                if missing_ch == 2:
                    if len(remain_eeg) > 0 and len(missing_eeg) > 0:
                        ref_eeg_idx = remain_eeg[0]
                        for me in missing_eeg:
                            full_data[me, :] = full_data[ref_eeg_idx, :]

                    if len(remain_eog) > 0 and len(missing_eog) > 0:
                        ref_eog_idx = remain_eog[0]
                        for me in missing_eog:
                            full_data[me, :] = full_data[ref_eog_idx, :]
                elif missing_ch == 1:
                    if len(remain_eeg) > 0 and len(missing_eeg) > 0:
                        eeg_mean = np.mean(full_data[remain_eeg, :], axis=0)  # shape=(n_times,)
                        for me in missing_eeg:
                            full_data[me, :] = eeg_mean

                    if len(remain_eog) > 0 and len(missing_eog) > 0:
                        ref_eog_idx = remain_eog[0]
                        for me in missing_eog:
                            full_data[me, :] = full_data[ref_eog_idx, :]

                final_ch_types = []
                for i, ch_name in enumerate(needed_channels):
                    if i in eeg_range:
                        final_ch_types.append('eeg')
                    elif i in eog_range:
                        final_ch_types.append('eog')
                    elif i in emg_range:
                        final_ch_types.append('emg')
                    else:
                        final_ch_types.append('misc')  # 혹은 필요에 따라

                info = mne.create_info(ch_names=needed_channels, sfreq=sfreq, ch_types=final_ch_types)
                info.set_meas_date(raw.info['meas_date'])
                new_raw = mne.io.RawArray(full_data, info, verbose=False)
                raw = new_raw

    if resample:
        try:
            mne.utils.set_config('MNE_USE_CUDA', 'true')
            mne.cuda.init_cuda(verbose=False)
            raw.resample(resample, n_jobs='cupy')
        except:
            raw.resample(resample)

    if exclude and preset:
        raw = raw.reorder_channels(PRESET[preset])

    return raw, n_missing_ch
