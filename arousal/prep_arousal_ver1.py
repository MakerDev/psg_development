import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pickle
import os
from os.path import basename, join 

from mne.filter import filter_data
from utils.tools import *

def band_pass_filter(x, lf, hf, fs=100, tb='auto'):

    return filter_data(
        x, sfreq=fs, l_freq=lf, h_freq=hf, verbose=False,
        l_trans_bandwidth=tb, h_trans_bandwidth=tb)

def prep_psg_signal(x, transpose=True, fs=100):
    eeg = np.array([
        band_pass_filter(x=x[i], lf=0.5, hf=24.999, fs=fs)
        for i in range(6)
    ])  # EEG
    eog = np.array([
        band_pass_filter(x=x[i], lf=0.5, hf=24.999, fs=fs)
        for i in range(6,8)
    ])  # EOG
    emg = band_pass_filter(x=x[-1], lf=0.5, hf=24.999, fs=fs)[np.newaxis,:]  # EMG

    x = np.vstack([eeg, eog, emg])

    center = np.mean(x, axis=1)
    scale = np.std(x, axis=1)
    scale[scale == 0] = 1.0
    x = (x.T - center) / scale
    
    if not transpose: 
        x = x.T

    return x


def process_edf_arousal_to_pickle(edf_path, xml_path, save_dir="./output", sfreq=100, xml_sample_dir=None, pad=True):
    # EDF 로드
    raw = load_edf_file(path=edf_path, preload=True, resample=sfreq, preset="STAGENET", exclude=True, missing_ch='handling')
    
    # Arousal XML 로드
    events = load_arousal_xml(xml_path)
    
    # EDF 시작 시간
    meas_date = raw.info['meas_date']
    data = raw.get_data()
    # 필터, 정규화
    x = prep_psg_signal(data, transpose=True, fs=sfreq)  # (time, ch)
    
    total_samples = x.shape[0]
    y = create_arousal_labels(events, meas_date, total_samples, sfreq=sfreq)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    if xml_sample_dir:
        xml_sample_path = join(xml_sample_dir, filename.replace(".pkl", "_AROUS.xml"))
        save_arousal_xml(meas_date, y, sfreq, xml_sample_path)

    if pad:
        max_len = 2**22 if sfreq == 100 else 2**21
        x, y = pad_signals(x, y, max_len)
        save_path = save_path.replace("AROUSAL", "AROUSAL_PAD")

    with open(save_path, "wb") as f:
        pickle.dump({"x": x, "y": y}, f)

    print("Saved:", save_path)


if __name__ == "__main__":
    sfreq = 50
    pad = True
    edf_dir = "/home/honeynaps/data/dataset/EDF"
    xml_dir = "/home/honeynaps/data/dataset/EBX/AROUS"
    pred_dir = "/home/honeynaps/data/dataset/EBX/AROUS_PRED"
    save_dir = "/home/honeynaps/data/dataset/PICKLE/AROUSAL_50"

    if pad:
        save_dir += "_PAD"

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]

    for edf_file in edf_files:
        edf_path = join(edf_dir, edf_file)
        xml_path = join(xml_dir, edf_file.replace(".edf", "_AROUS.xml"))

        try:
            process_edf_arousal_to_pickle(edf_path, xml_path, save_dir, sfreq, pred_dir, pad)
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            continue
