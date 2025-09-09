# %%
from scipy.signal import butter, lfilter
from tsflex.processing import SeriesPipeline, SeriesProcessor

# %%
def butter_bandpass_filter(sig, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, sig)
    return y

eeg_bandpass = SeriesProcessor(
    function=butter_bandpass_filter,
    series_names=["EEG1", "EEG2", "EEG3", "EEG4", "EEG5",  "EEG6", "EOG1", "EOG2"],
    lowcut=0.4,
    highcut=30,
    fs=100,
)

emg_bandpass = SeriesProcessor(
    function=butter_bandpass_filter,
    series_names=["EMG"],
    lowcut=0.5,
    highcut=10,
    fs=100,
)

process_pipe = SeriesPipeline(
    [
        eeg_bandpass,
        emg_bandpass,
    ]
)
channel_names = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5",  "EEG6", "EOG1", "EOG2", "EMG"]


# %%
tsfresh_settings = {
    "fft_aggregated": [
        {"aggtype": "centroid"},
        {"aggtype": "variance"},
        {"aggtype": "skew"},
        {"aggtype": "kurtosis"},
    ],
    "fourier_entropy": [
        {"bins": 2},
        {"bins": 3},
        {"bins": 5},
        {"bins": 10},
        {"bins": 30},
        {"bins": 60},
        {"bins": 100},
    ],
    "binned_entropy": [
        {"max_bins": 5},
        {"max_bins": 10},
        {"max_bins": 30},
        {"max_bins": 60},
    ],
}


# %%
import numpy as np
import antropy as ant
import scipy.stats as ss
from yasa import bandpower

import scipy.stats as ss
from tsflex.features import (
    FeatureCollection,
    FuncWrapper,
    MultipleFeatureDescriptors,
    FuncWrapper,
)
from tsflex.features.integrations import tsfresh_settings_wrapper


def wrapped_higuchi_fd(x):
    x = np.array(x, dtype="float64")
    return ant.higuchi_fd(x)


bands = [
    (0.4, 1, "sdelta"),
    (1, 4, "fdelta"),
    (4, 8, "theta"),
    (8, 12, "alpha"),
    (12, 16, "sigma"),
    (16, 30, "beta"),
]
bandpowers_ouputs = [b[2] for b in bands] + ["TotalAbsPow"]


def wrapped_bandpowers(x, sf, bands):
    return bandpower(x, sf=sf, bands=bands).values[0][:-2]


time_funcs = [
    np.std,
    ss.iqr,
    ss.skew,
    ss.kurtosis,
    ant.num_zerocross,
    FuncWrapper(
        ant.hjorth_params, output_names=["horth_mobility", "hjorth_complexity"]
    ),
    wrapped_higuchi_fd,
    ant.petrosian_fd,
    ant.perm_entropy,
] + tsfresh_settings_wrapper(tsfresh_settings)

sf = 100
freq_funcs = [
    FuncWrapper(wrapped_bandpowers, sf=sf, bands=bands, output_names=bandpowers_ouputs)
]

channel_names = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5",  "EEG6", "EOG1", "EOG2", "EMG"]

time_feats = MultipleFeatureDescriptors(
    time_funcs,
    channel_names,
    windows=["30s", "60s", "90s"],
    strides="30s",
)
freq_feats = MultipleFeatureDescriptors(
    freq_funcs,
    channel_names[:-1],
    windows=["30s", "60s", "90s"],
    strides="30s",
)

feature_collection = FeatureCollection([time_feats, freq_feats])

# %%
import pickle
import pandas as pd

def load_features(file_path, save_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        x = data["x"] # N, 3000, 9
        y = data["y"] # N

    # Leave only y != -1
    mask = y != -1
    x = x[mask]
    y = y[mask]

    # x = x.reshape(-1, x.shape[2]).T
    x = x.reshape(-1, x.shape[2])
    data = []
    # Create a time index (start_time is not needed, use a generic range)
    # Assume sampling frequency is 100Hz (1 sample per 0.01 seconds)
    time_index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="10ms")

    for i, channel_name in enumerate(channel_names):
        data.append(pd.Series(x[:, i], index=time_index, name=channel_name))

    feats = feature_collection.calculate(data, return_df=True, show_progress=False)

    if feats.shape[0] != y.shape[0]:
        diff = y.shape[0] - feats.shape[0]
        # y = y[diff:diff + feats.shape[0]]
        y = y[:feats.shape[0]] # 이게 나음 결과상으로. 얘가 맞는듯

    no_shift_cols = [c for c in feats.columns if not "shift=" in c]# or "w=1m" in c or "w=1m30s" in c]
    # len(no_shift_cols)

    normal_nan_mask = feats[no_shift_cols].isna().sum() == 0
    feats[np.array(no_shift_cols)[~normal_nan_mask]].isna().sum().sort_values()[::-1]
    feats[np.array(no_shift_cols)[~normal_nan_mask]].isna().any(axis=1).sum() / len(feats)

    feats_np = feats.to_numpy()
    nans = []

    for i in range(feats_np.shape[0]):
        for j in range(feats_np.shape[1]):
            if np.isnan(feats_np[i, j]):
                nans.append(i)
                break
    # Remove the rows with NaN values
    feats_np = np.delete(feats_np, nans, axis=0)
    y = np.delete(y, nans, axis=0)


    print(feats_np.shape, y.shape)
    with open(save_path, "wb") as f:
        pickle.dump({"x": feats_np, "y": y}, f)
        
import os

# prep_window_wise.py를 통해 만들어진 pickle 파일을 feature로 변환
folder_name = "SLEEP_50_NOFILL"
file_dir = f"/home/honeynaps/data/dataset/PICKLE/{folder_name}"
save_dir = f"/home/honeynaps/data/dataset/FEATURES/{folder_name}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

files = os.listdir(file_dir)

for file in files:
    file_path = os.path.join(file_dir, file)
    save_path = os.path.join(save_dir, file)

    if os.path.exists(save_path):
        print(file, "already done")

    load_features(file_path, save_path)
    print(file, "done")

