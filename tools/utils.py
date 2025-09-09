import xml.etree.ElementTree as ET
import numpy as np
import datetime as dt
import argparse


SLEEPSTAGE_TO_LABEL = {
    "SLEEP-U":-1,
    "SLEEP-W":0, 
    "SLEEP-R":1,
    "SLEEP-1":2, 
    "SLEEP-2":3, 
    "SLEEP-3":4,
    "SLEEP-WAKE":0,
    "SLEEP-REM":1,
    "SLEEP-N1":2,
    "SLEEP-N2":3,
    "SLEEP-N3":4,
}

def str2bool(v):
    """문자열 형태의 인자를 bool 값으로 변환하기 위한 헬퍼 함수"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def load_sleep_stage(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    first_onset = None
    events = []
    for i, annotation in enumerate(root.findall("annotation")):
        onset = annotation.find("onset").text
        if i == 0:
            first_onset = onset
        duration = float(annotation.find("duration").text)
        description = annotation.find("description").text
        onset = dt.datetime.strptime(onset,"%Y-%m-%dT%H:%M:%S.%f")
        events.append(SLEEPSTAGE_TO_LABEL[description])  # Default to -1 if not found

    return events, first_onset