import warnings
import argparse
warnings.filterwarnings('ignore')

from datetime import timedelta
import xml.etree.ElementTree as ET
from datetime import timedelta
import mne 
from mne.io import read_raw_edf
from os.path import basename
from .config import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_edf_file(path, preload=False, resample=100, preset="STAGENET", exclude=True, missing_ch='raise'):
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

    if resample:
        try:
            mne.utils.set_config('MNE_USE_CUDA', 'true')  
            mne.cuda.init_cuda(verbose=False)
            raw.resample(resample, n_jobs='cupy')
        except:
            raw.resample(resample)

    if exclude and preset:
        raw = raw.reorder_channels(PRESET[preset])

    return raw


def save_sleepstage_xml(meas_date, y, xml_save_path, location="EEG-F4", probs=None):
    label_to_stage = {
        0: "SLEEP-W",
        1: "SLEEP-R",
        2: "SLEEP-1",
        3: "SLEEP-2",
        4: "SLEEP-3"
    }

    root = ET.Element("annotationlist")

    for i, stage_val in enumerate(y):
        start_sec = i * 30
        onset_time = meas_date + timedelta(seconds=start_sec)

        duration = 30.0

        description = label_to_stage.get(stage_val, "SLEEP-U")

        annotation = ET.SubElement(root, "annotation")

        # onset
        onset_elem = ET.SubElement(annotation, "onset")
        onset_elem.text = onset_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        # duration
        duration_elem = ET.SubElement(annotation, "duration")
        duration_elem.text = f"{duration:.6f}"

        # description
        desc_elem = ET.SubElement(annotation, "description")
        desc_elem.text = description

        # location
        location_elem = ET.SubElement(annotation, "location")
        location_elem.text = location

        if probs is not None:
            pred_prob_elem = ET.SubElement(annotation, "probability")
            pred_prob_elem.text = f"{probs[i][stage_val]:.4f}"

    # XML tree 작성
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)

    # XML 파일로 저장
    tree.write(xml_save_path, encoding="UTF-8", xml_declaration=True)

if __name__ == '__main__':
    raw = load_edf_file("/home/honeynaps/data/GOLDEN/EDF2/SCH-230114R3_M-60-OV-SE.edf", 
                        preload=True, preset="STAGENET", exclude=False)
    print(raw.ch_names)
    