import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import argparse
from os import path
from datetime import timedelta
import datetime as dt
import xml.etree.ElementTree as ET

import mne 
from mne.io import read_raw_edf
from os.path import basename

from datetime import datetime, timedelta
from mne.filter import filter_data
from .config import *
import mne
from mne.preprocessing import ICA


def remove_artifacts_ica(raw, 
                         n_components=20, 
                         method='fastica', 
                         eog_ch=None, 
                         ecg_ch=None):
    # 1) ICA 객체 생성
    ica = ICA(n_components=n_components, method=method, random_state=42)
    
    # 2) EEG + EOG + ECG 채널 골라서 ICA 학습
    #    굳이 EMG나 STIM 등은 제외
    #    => picks='all' 해도 되지만, 데이터가 너무 많으면 시간 증가
    raw_ica = raw.copy().pick(eeg=True, eog=True, ecg=True, exclude='bads')
    ica.fit(raw_ica)

    # 3) EOG 아티팩트 탐지
    eog_inds, eog_scores = [], []
    if eog_ch is not None:
        eog_inds, eog_scores = ica.find_bads_eog(raw_ica, ch_name=eog_ch)
        ica.exclude.extend(eog_inds)

    # 4) ECG 아티팩트 탐지
    ecg_inds, ecg_scores = [], []
    if ecg_ch is not None:
        ecg_inds, ecg_scores = ica.find_bads_(raw_ica, ch_name=ecg_ch)
        ica.exclude.extend(ecg_inds)

    # 예) 나머지 아티팩트(근육 EMG 등)은 수동으로 확인할 수도 있음
    #  -> ica.plot_components(), ica.plot_properties(...) etc.

    # 5) ICA 적용 (제외된 성분들 뺀 상태로)
    ica.apply(raw)
    return raw




def load_edf_only(edf_path, prep_fn, start_time=None, sfreq=100, pad=True, prep_fn_args={}):
    raw = load_edf_file(path=edf_path, preload=True, resample=sfreq, preset="STAGENET", exclude=True, missing_ch='handling')
    meas_date = raw.info['meas_date'].replace(tzinfo=None)

    if start_time is None:
        start_time = meas_date
    else:
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=None)
    start_sec = (start_time - meas_date).total_seconds()
    start_idx = int(start_sec * sfreq)

    data = raw.get_data()

    x = prep_fn(data, **prep_fn_args)
    x = x[start_idx:]
    
    if pad:
        max_len = 2**22 if sfreq == 100 else 2**21
        x, y = pad_signals(x, np.zeros(x.shape[0]), max_len)
    
    return x, y


def load_edf_file(path, preload=False, resample=100, preset="STAGENET", exclude=True, missing_ch='raise', do_filter=True, ica_remove=False):
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

    try:
        if exclude and preset:
            raw.pick(PRESET[preset])
    except Exception as e:
        raise Exception(str(e) + f" C_NAMES: {str(raw.ch_names)}")

    if missing_ch == 'raise' and exclude:
        if set(PRESET[preset]) != set(raw.ch_names):
            raise RuntimeError('{0} file doesn\'t satisfied with PRESET {1}\nPRESET KEYS: {2}\nEDF KEYS   : {3}'.format(
                basename(path), preset, PRESET[preset], raw.ch_names
            ))

    elif missing_ch == 'handling' and exclude:
        target_key = list(set(PRESET[preset])-set(raw.ch_names))
        replace_map = replace_key_mapping(target_key, raw.ch_names)

        ch_names = [ key for key in replace_map ] + raw.ch_names
        ch_picks = [ raw.ch_names.index(ch) for ch in replace_map.values() ] + [ raw.ch_names.index(ch) for ch in raw.ch_names ]
        ch_types = ['mag' for _ in ch_names]

        if set(PRESET[preset]) != set(ch_names):
            raise RuntimeError('{0} file doesn\'t satisfied with PRESET {1}\nPRESET KEYS: {2}\nEDF KEYS   : {3}'.format(
                basename(path), preset, PRESET[preset], ch_names
            ))

        info = mne.create_info(ch_names=ch_names, sfreq=raw.info['sfreq'], ch_types=ch_types)
        info.set_meas_date(raw.info['meas_date'])
        data = raw.get_data(picks=ch_picks)
        raw = mne.io.RawArray(data, info, verbose=False)

    # raw.set_eeg_reference('average', projection=False, verbose=False)
    if do_filter:
        raw.filter(l_freq=0.3, h_freq=35, picks='all', method='fir', 
                   fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=[50.0], picks='all', verbose=False)

    if resample:
        try:
            mne.utils.set_config('MNE_USE_CUDA', 'true')  
            mne.cuda.init_cuda(verbose=False)
            raw.resample(resample, n_jobs='cupy')
        except:
            raw.resample(resample)

    if exclude and preset:
        raw = raw.reorder_channels(PRESET[preset])

    if ica_remove:
        eog_ch_name = None
        ecg_ch_name = None


        possible_eog = [ch for ch in raw.ch_names if 'EOG' in ch.upper()]
        if len(possible_eog) > 0:
            eog_ch_name = possible_eog[0]

        possible_ecg = [ch for ch in raw.ch_names if 'ECG' in ch.upper()]
        if len(possible_ecg) > 0:
            ecg_ch_name = possible_ecg[0]
        
        raw = remove_artifacts_ica(raw, 
                                   n_components=20,
                                   method='fastica',
                                   eog_ch=eog_ch_name,
                                   ecg_ch=ecg_ch_name)


    return raw


def load_arousal_xml_shhs(xml_path: str):
    """
    Parse a Compumedics PSGAnnotation file and return a list of events.

    Each element in the returned list has the keys:
        - start      : onset time in seconds from recording start
        - duration   : event duration in seconds (float)
        - description: text from <EventConcept>
        - location   : signal channel name from <SignalLocation> (may be None)
        - event_type : text from <EventType>               (optional, can help post-filtering)

    Parameters
    ----------
    xml_path : str
        Path to the .xml file.

    Returns
    -------
    List[Dict[str, Any]]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    events = []

    # Optional: capture absolute recording start clock time for later use
    # (the very first ScoredEvent usually stores it)
    abs_recording_start = None
    for se in root.find("ScoredEvents").findall("ScoredEvent"):
        concept = se.findtext("EventConcept", default="")
        if "Recording Start Time" in concept:
            clock_str = se.findtext("ClockTime")         # e.g. "00.00.00 22.00.00"
            # Compumedics format varies; keep raw string for reference
            abs_recording_start = clock_str
            break

    # Iterate over all scored events
    for se in root.find("ScoredEvents").findall("ScoredEvent"):
        # Skip the "Recording Start Time" marker itself
        if "Recording Start Time" in se.findtext("EventConcept", default=""):
            continue

        start_sec = float(se.findtext("Start", default="nan"))
        duration_sec = float(se.findtext("Duration", default="nan"))
        description = se.findtext("EventConcept", default=None).lower()
        location = se.findtext("SignalLocation", default=None)
        event_type = se.findtext("EventType", default=None)

        if "arous" not in description:
            continue

        events.append(
            {
                "start": start_sec,
                "duration": duration_sec,
                "description": description,
                "location": location,
                "event_type": event_type,
                # "start_time": convert_to_datetime(abs_recording_start, start_sec)  # if you later implement this
            }
        )

    return events

def load_arousal_xml(xml_path):
    """Load the arousal XML and return a list of events with onset, duration, description"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    events = []
    for annotation in root.findall("annotation"):
        onset = annotation.find("onset").text
        duration = float(annotation.find("duration").text)
        description = annotation.find("description").text
        location = annotation.find("location").text

        events.append({
            "onset": dt.datetime.strptime(onset,"%Y-%m-%dT%H:%M:%S.%f"),
            "duration": duration,
            "description": description,
            "location": location
        })
    return events


def load_sleep_stage(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    events = []
    for annotation in root.findall("annotation"):
        onset = annotation.find("onset").text
        duration = float(annotation.find("duration").text)
        description = annotation.find("description").text
        onset = dt.datetime.strptime(onset,"%Y-%m-%dT%H:%M:%S.%f")
        events.append({
            "onset": onset,
            "duration": duration,
            "description": description
        })
    return events

def create_arousal_labels(events, meas_date, total_samples, sfreq=100):
    y = np.zeros(total_samples, dtype=np.float32)
    meas_date = meas_date.replace(tzinfo=None)
    for event in events:
        start_sec = (event["onset"] - meas_date).total_seconds()
        end_sec = start_sec + event["duration"]
        start_idx = max(int(start_sec * sfreq), 0)
        end_idx = min(int(end_sec * sfreq), total_samples)
        y[start_idx:end_idx] = 1.0
    return y


def pad_signals(x, y, max_len=2**22):
    x = np.transpose(x, (1, 0))
    curr_len = x.shape[1]
    padd = max_len - curr_len
    if padd > 0:
        left_pad = padd // 2 + padd % 2
        right_pad = padd // 2

        x = np.pad(x, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)
        y = np.pad(y, (left_pad, right_pad), mode='constant', constant_values=-1)

    assert x.shape[1] == max_len

    return x, y

def save_arousal_xml(meas_date, y, sfreq, xml_save_path, description="AROUS", location="EEG-F3", min_duration=3):
    diff_y = np.diff(np.concatenate([[0], y, [0]]))  
    start_points = np.where(diff_y == 1)[0]
    end_points = np.where(diff_y == -1)[0]

    root = ET.Element("annotationlist")

    for start_idx, end_idx in zip(start_points, end_points):
        start_sec = start_idx / sfreq
        end_sec = end_idx / sfreq
        duration = end_sec - start_sec

        if duration < min_duration:
            continue

        onset_time = meas_date + timedelta(seconds=start_sec)

        annotation = ET.SubElement(root, "annotation")

        onset_elem = ET.SubElement(annotation, "onset")
        onset_elem.text = onset_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        duration_elem = ET.SubElement(annotation, "duration")
        duration_elem.text = f"{duration:.6f}"

        desc_elem = ET.SubElement(annotation, "description")
        desc_elem.text = description

        location_elem = ET.SubElement(annotation, "location")
        location_elem.text = location

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)

    tree.write(xml_save_path, encoding="UTF-8", xml_declaration=True)