import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import argparse
from os import path
from datetime import timedelta
import pandas as pd
import pickle
import os
import datetime as dt
import xml.etree.ElementTree as ET
import pyedflib
from xml.dom import minidom
from datetime import datetime, timedelta

import mne 
from mne.io import read_raw_edf
from os.path import basename, join 

from scipy.signal import find_peaks, hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from mne.filter import filter_data
from utils.tools import *



def get_events_from_labels(labels):
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(labels):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            end = i-1
            events.append((start, end))
    if in_event:
        events.append((start, len(labels)-1))
    return events


def robust_scale(x):
    # x: (channels, time)
    median = np.median(x, axis=1, keepdims=True)
    mad    = np.median(np.abs(x - median), axis=1, keepdims=True) + 1e-9
    return (x - median) / mad

def detect_artifact_mask(data, flat_thresh=1e-6, spike_thresh=200e-6):
    # data: (channels, time)
    flat_mask  = np.all(np.abs(data) < flat_thresh, axis=0)
    spike_mask = np.any(np.abs(data) > spike_thresh, axis=0)
    return (flat_mask | spike_mask).astype(np.uint8)  # 0/1 mask


def process_edf_arousal_to_pickle(edf_path, xml_path, save_dir="./output", sfreq=100):
    raw = load_edf_file(path=edf_path, preload=True, resample=sfreq, preset="STAGENET", exclude=True, missing_ch='handling')
    
    events = load_arousal_xml(xml_path)
    
    meas_date = raw.info['meas_date']
    data = raw.get_data()

    x = np.array(data)
    art_mask = detect_artifact_mask(x)
    art_mask = art_mask[np.newaxis, :]
    x = robust_scale(data)

    print("Masked ", np.sum(art_mask)//sfreq, "secs")
    
    total_samples = x.shape[1]
    y = create_arousal_labels(events, meas_date, total_samples, sfreq=sfreq)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = basename(edf_path).replace(".edf", ".pkl")
    save_path = join(save_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump({"x": x, 
                     "y": y,
                     "art_mask": art_mask,
                     "meas_date": meas_date,}, 
                     f)

    print("Saved:", save_path)




if __name__ == "__main__":
    sfreq = 50
    
    base_dir = "/home/honeynaps/data/GOLDEN"
    edf_dir = f"{base_dir}/EDF2"
    xml_dir = f"{base_dir}/EBX2/AROUS"

    base_dir = "/home/honeynaps/data/dataset2"
    edf_dir = f"{base_dir}/EDF"
    xml_dir = f"{base_dir}/EBX/AROUS"

    base_dir = "/home/honeynaps/data/HN_DATA_AS"
    edf_dir = f"{base_dir}/EDF"
    xml_dir = f"{base_dir}/EBX/ASHIFT"

    base_dir = "/home/honeynaps/data/GOLDEN2"
    edf_dir = f"{base_dir}/EDF"
    xml_dir = f"{base_dir}/TRUE/AROUS"
    
    save_dir = f"{base_dir}/PICKLE/AROUSAL_TIME_{sfreq}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")]
    swap = "AROUS" if "HN_DATA" not in base_dir else "ASHIFT"

    for i, edf_file in enumerate(edf_files):
        edf_path = join(edf_dir, edf_file)
        xml_path = os.path.join(xml_dir, edf_file.replace(".edf", f"_{swap}.xml"))

        try:
            process_edf_arousal_to_pickle(edf_path, xml_path, 
                                          save_dir, sfreq)
            print(f"Done processing {i+1}/{len(edf_files)}: {edf_file}")
        except Exception as e:
            print(f"Error: {edf_file}, {str(e)}")
            continue
