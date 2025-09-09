import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import timedelta
import datetime as dt
import xml.etree.ElementTree as ET

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

def save_micro_events_by_channels(meas_date, y_by_channels, sfreq, xml_save_path, description="KCOMP", min_duration=3):
    root = ET.Element("annotationlist")
    meas_date = meas_date.replace(tzinfo=None)
    recording_duration = len(next(iter(y_by_channels.values()))) // sfreq
    ET.SubElement(root, "recording_start_time").text = meas_date.strftime("%Y-%m-%dT%H:%M:%S.%f")
    ET.SubElement(root, "recording_duration").text = f"{recording_duration}"

    for channel, y in y_by_channels.items():
        append_to_root(root, meas_date, y, sfreq, channel, 
                       description=description, min_duration=min_duration)
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)

    with open(xml_save_path, "wb") as fp:
        fp.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(fp, encoding="UTF-8", xml_declaration=False)

def save_micro_events_by_channels_and_type(meas_date, 
                                           y_by_channels_by_pred_type, 
                                           sfreq, xml_save_path, 
                                           description="KCOMP", min_duration=3):
    root = ET.Element("annotationlist")
    for pred_type, y_by_channels in y_by_channels_by_pred_type.items():
        for channel, y in y_by_channels.items():
            append_to_root(root, meas_date, y, sfreq, channel, 
                           description=f"{description}-{pred_type}", min_duration=min_duration)
            
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)

    with open(xml_save_path, "wb") as fp:
        fp.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(fp, encoding="UTF-8", xml_declaration=False)

def append_to_root(root, meas_date, y, sfreq, channel, description="KCOMP", min_duration=3, 
                   add_channel_postfix=False):
    diff_y = np.diff(np.concatenate([[0], y, [0]]))
    start_points = np.where(diff_y == 1)[0]
    end_points = np.where(diff_y == -1)[0]

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
        if add_channel_postfix:
            desc_elem.text = f"{description}-{channel}"
        else:
            desc_elem.text = description

        location_elem = ET.SubElement(annotation, "location")
        location_elem.text = f"EEG-{channel}"
    
def rebuild_label_xml(xml_path, new_xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for annotation in root.findall("annotation"):
        description = annotation.find("description")
        location = annotation.find("location")
        if description is not None:
            new_desc = f"{description.text}-{location.text.split('EEG-')[1]}"
            description.text = new_desc.replace("MW_EEG-", "LBL-")

    with open(new_xml_path, "wb") as fp:
        fp.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(fp, encoding="UTF-8", xml_declaration=False)

if __name__ == "__main__":
    old_xml_dir = '/home/honeynaps/data/HN_DATA_MW/EBX/MW_EEG_NEW_ALL'
    new_xml_dir = '/home/honeynaps/data/HN_DATA_MW/EBX/MW_EEG_VIS'

    import os
    for file in os.listdir(old_xml_dir):
        if file.endswith('.xml'):
            old_xml_path = os.path.join(old_xml_dir, file)
            new_xml_path = os.path.join(new_xml_dir, file.replace('.xml', '_VIS.xml'))
            rebuild_label_xml(old_xml_path, new_xml_path)
            print(f"Rebuilt {file} and saved to {new_xml_path}")