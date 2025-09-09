import mne
import numpy as np
import xml.etree.ElementTree as ET
import datetime as dt
import pickle
import os

from scipy.signal import spectrogram

def stage_to_int(stage_str):
    mapping = {
        'SLEEP-W': 0,
        'SLEEP-1': 1,
        'SLEEP-2': 2,
        'SLEEP-3': 3,
        'SLEEP-R': 4
    }
    return mapping.get(stage_str, -1)

def parse_sleep_stages(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ann_list = []
    for ann in root.findall('annotation'):
        onset_str = ann.find('onset').text
        duration_str = ann.find('duration').text
        desc = ann.find('description').text
        
        dt_fmt = "%Y-%m-%dT%H:%M:%S.%f"
        try:
            onset_dt = dt.datetime.strptime(onset_str, dt_fmt)
        except:
            dt_fmt = "%Y-%m-%dT%H:%M:%S"
            onset_dt = dt.datetime.strptime(onset_str, dt_fmt)

        dur = float(duration_str)
        ann_list.append({
            'onset_dt': onset_dt,
            'duration': dur,
            'stage': desc
        })
    ann_list.sort(key=lambda x: x['onset_dt'])
    return ann_list


def make_spectrogram(data_2d, fs=50, nperseg=100, noverlap=50):
    T, C = data_2d.shape
    specs = []
    for ch in range(C):
        f, t, Sxx = spectrogram(
            data_2d[:, ch], fs=fs,
            window='hann',
            nperseg=nperseg, noverlap=noverlap,
            nfft=nperseg, scaling='density', mode='psd'
        )
        Sxx_log = np.log1p(Sxx)
        specs.append(Sxx_log[None, ...])
    spec = np.concatenate(specs, axis=0)
    return spec, f, t

def process_edf_sleep_stage_spect(edf_path, xml_path, save_path, fs=50,
                                  nperseg=100, noverlap=50):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(fs)
    data = raw.get_data()  # (channels, samples)
    meas_date = raw.info['meas_date']
    if meas_date is None:
        meas_date = dt.datetime(1970,1,1)

    n_ch, n_samp = data.shape
    stage_list = parse_sleep_stages(xml_path)

    X_list, Y_list = [], []
    freq_array = None
    time_array = None

    for ann in stage_list:
        onset_dt = ann['onset_dt']
        stage_str = ann['stage']
        label_int = stage_to_int(stage_str)
        if label_int < 0:  # unknown stage
            continue

        # compute start sample
        event_start_sec = (onset_dt.replace(tzinfo=None) - meas_date.replace(tzinfo=None)).total_seconds()
        s_idx = int(round(event_start_sec * fs))
        e_idx = s_idx + 30*fs  # 30초 길이

        if s_idx < 0 or e_idx > n_samp:
            continue
        
        epoch_data = data[:, s_idx:e_idx]  # (n_ch, 30*fs)
        epoch_data_2d = epoch_data.T       # (30*fs, n_ch)

        spec, freqs, t_bins = make_spectrogram(epoch_data_2d, fs=fs,
                                               nperseg=nperseg, noverlap=noverlap)
        # spec: (n_ch, freq, time_bins)
        X_list.append(spec.astype(np.float32))
        Y_list.append(label_int)

        if freq_array is None:
            freq_array = freqs
            time_array = t_bins

    if len(X_list) == 0:
        print("No epochs found. skip.")
        return

    X = np.stack(X_list, axis=0)  # (n_epochs, n_ch, freq, time_bins)
    Y = np.array(Y_list, dtype=np.int32)

    out_dict = {
        "X": X,  # (n_epochs, 9, freq, time)
        "Y": Y,  # (n_epochs,)
        "freqs": freq_array,
        "t_bins": time_array,
        "fs": fs,
        "meas_date": meas_date
    }
    with open(save_path, "wb") as f:
        pickle.dump(out_dict, f)
    print(f"Saved {save_path}, X={X.shape}, Y={Y.shape}")

# ---------- main test -----------
if __name__ == "__main__":
    edf_dir = "/home/honeynaps/data/GOLDEN/EDF2"
    xml_dir = "/home/honeynaps/data/GOLDEN/EBX2/SLEEP"
    fs = 50
    out_dir = f"/home/honeynaps/data/GOLDEN/SPEC/SLEEP_{fs}"
    os.makedirs(out_dir, exist_ok=True)

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]
    for edf_file in edf_files:
        edf_path = os.path.join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf","_SLEEP.xml"))
        save_path = os.path.join(out_dir, edf_file.replace(".edf",".pkl"))
        process_edf_sleep_stage_spect(edf_path, xml_path, save_path,
                                      fs=50, nperseg=50, noverlap=25)
