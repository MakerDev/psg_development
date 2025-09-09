#%% 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import argparse
from os import path
from datetime import datetime, timedelta
import pickle
import os

from models.stagenet import *
from modules.iofiles import edf, config
from modules.dataset       import SCH
from modules.preprocessing import prep_psg_signal
from modules.iofiles import edf as edf_io


# Config DEMO 
batch_size = 512

def epoching_with_events(x, events, sfreq=50, dtype=np.float32):
    event_map = config.EVENT_MAP["STAGENET"]

    X, Y = [], []
    
    for event in events[:-2]:
        st, et = int(event['s_sec']*sfreq), int(event['e_sec']*sfreq)
        epoch_x = x[st:et]
        epoch_y = event_map[event['annotation']]
        X.append(epoch_x); Y.append(epoch_y)

    X = np.array(X).astype(dtype)
    Y = np.array(Y).astype(dtype)
    return X, Y

def epoching_from_time(x, base_time, start_time, sfreq=50, dtype=np.float32, window_size=30):
    X = []

    if x.shape[0] < window_size*sfreq:
        x = x.T

    start_idx = int((start_time - base_time).total_seconds() * sfreq)
    
    for i in range(start_idx, len(x) - window_size*sfreq + 1, window_size*sfreq):
        epoch_x = x[i:i+window_size*sfreq]
        X.append(epoch_x)

    X = np.array(X).astype(dtype)

    return X


def load_edf(path_edf:str):
    """[summary]
    Load edf file. this function have dependancy with honeytools and mne. 

    Arguments:
        path_edf {str} -- EDF file path.
    """
    raw = edf.load(
        path=path_edf, preload=True, resample=50, 
        preset="STAGENET", exclude=True, missing_ch='handling')

    base_time = raw.info['meas_date']
    x = prep_psg_signal(raw.get_data())
    x.reshape()

    return x, base_time

def load_xml_edf(path_edf:str, path_xml:str, sfreq:int=100, t:str='SCH', fill_na:bool=True, save_dir=None):
    assert t in ['SCH'], "Choose -t in ['SCH']"
    dir_edf, file_edf = path.split(path_edf)
    dir_xml, file_xml = path.split(path_xml)

    save_dir = f"/home/honeynaps/data/250718_CND/PICKLE/SLEEP_{sfreq}"
    if not fill_na:
        save_dir += "_NOFILL_NEWPREP"

    if not path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, file_edf.replace("edf", "pickle"))

    # if os.path.exists(save_path):
    #     print('Skipping', file_edf)
    #     return False, ""

    def build_key(file_edf, file_xml):

        key, length = '', min([len(file_edf), len(file_xml)])
        for ch1, ch2 in zip(file_edf[:length], file_xml[:length]):
            if ch1 == ch2: key += ch1 
            else: break
            
        return key

    key = build_key(file_edf, file_xml)

    params = {
        'root_edf'     : dir_edf,
        'root_xml'     : dir_xml,
        'ext_x'        : file_edf[len(key):],
        'ext_y'        : file_xml[len(key):],
        'missing_ch'   : 'raise', 
        'multi_buffer' : True }
    
    dataset = SCH(**params)
    
    try:
        edf = dataset.load_edf(key, preload=True, resample=sfreq, 
                preset="STAGENET", exclude=True)
        xml = dataset.load_events(key, "STAGENET", fill_na=fill_na)
    except Exception as e:
        return True, str(e).split("\n")[0]

    st, et = min([ e['s_sec'] for e in xml ]), max([ e['e_sec'] for e in xml ])
    base_time = edf.info['meas_date'] + timedelta(seconds=st)
    
    data = prep_psg_signal(edf.get_data(), transpose=True, fs=sfreq)
    # data = edf.get_data()

    X, Y = epoching_with_events(data, xml, sfreq=sfreq)

    with open(save_path, 'wb') as f:
        pickle.dump({"x": X, "y": Y}, f)
    print("Successfully finished", file_edf)
    return False, ""



def initialize_demo(filename = "AJD-120120R1_F-20-OB-NO"):
    "Initialize Task with arguments"
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str,
                metavar='EDF path', default=f"/home/honeynaps/data/250718_CND/EDF2/{filename}.edf",
                help="EDF file path.")
    parser.add_argument('--xml', type=str, default=f'/home/honeynaps/data/250718_CND/EBX/SLEEP/{filename}_SLEEP.xml',
                help="XML file path.")
    parser.add_argument('--save', type=str, default='./',
                help="Directory path to save result xml file.")
    parser.add_argument('--f', type=str, default='SCH',
                choices=['SCH','SHHS'],
                help="Where the EDF file came from? SCH or SHHS")
    parser.add_argument('--verbose', type=bool, default=False,
                choices=[True,False],
                help="Will you print out process status?")
    parser.add_argument('--device', type=str, default='cuda:0',
                choices=['cpu', 'cuda:0','cuda:1','cuda:2','cuda:3'],
                help="Will you print out process status?")
    parser.add_argument('--mode', type=str, default="demo",
                choices=["demo","batch"],
                help="Tell me your purpose of execution.")


    "Parse task arguments"
    args = parser.parse_args()

    path_edf  = path.abspath(args.edf)
    path_xml  = path.abspath(args.xml) if args.xml != 'None' else None
    path_save = args.save
    verbose   = args.verbose
    device    = args.device
    mode      = args.mode

    "Asserting arguments"
    assert path.isfile(path_edf), f'{path_edf} is not file.'
    assert (path_xml==None or path.isfile(path_xml)), f'{path_xml} is not file.'
    assert path.isdir(path_save), f'{path_save} is not directory.'

    tag = args.f 
    if mode == 'batch':
        path_save = path.abspath(path.join(
            path_save, path.basename(path_edf).replace('.edf','.csv')))
    else:
        path_save = path.abspath(path.join(
            path_save, path.basename(path_edf).replace('.edf','_sn.xml')))

    "Print out init results"
    if verbose:
        print(f"[StageNet R2 Confidence Demo version 0.0.1]\n")
        print(f"EDF file: {path_edf}")
        print(f"XML file: {path_xml}")
        print(f"EDF from: {tag}\n")
        print(f"Results file will be saved at\n- {path_save}\n")

    return {
        'path_edf'  :path_edf,
        'path_xml'  :path_xml,
        'path_save' :path_save,
        'verbose'   :verbose,
        'device'    :device,
        'mode'      :mode }

def main(filename, fill_na=True, sfreq=100):
    params = initialize_demo(filename)
    is_xml = True if params['path_xml'] != None else False

    if not is_xml:
        return True, "No XML"

    # load edf data
    has_error, message = load_xml_edf(path_edf=params['path_edf'], path_xml=params['path_xml'], fill_na=fill_na, sfreq=sfreq)

    return has_error, message

def load_edf_for_demo(path_edf:str, path_xml:str, sfreq:int=50, t:str='SCH', fill_na:bool=False, missing_ch='raise'):
    assert missing_ch in ['raise', 0, 1, 2], "Choose missing_ch in ['raise', 0, 1, 2]"
    assert t in ['SCH', 'SHHS'], "Choose -t in ['SCH', 'SHHS']"
    dir_edf, file_edf = path.split(path_edf)
    dir_xml, file_xml = path.split(path_xml)

    def build_key(file_edf, file_xml):
        key, length = '', min([len(file_edf), len(file_xml)])
        for ch1, ch2 in zip(file_edf[:length], file_xml[:length]):
            if ch1 == ch2: key += ch1 
            else: break
            
        return key

    key = build_key(file_edf, file_xml)

    params = {
        'root_edf'     : dir_edf,
        'root_xml'     : dir_xml,
        'ext_x'        : file_edf[len(key):],
        'ext_y'        : file_xml[len(key):],
        'missing_ch'   : missing_ch, 
        'multi_buffer' : True }
    
    dataset = SCH(**params)
    
    try:
        edf = dataset.load_edf(key, preload=True, resample=sfreq, 
                preset="STAGENET", exclude=True)
        xml = dataset.load_events(key, "STAGENET", fill_na=fill_na)
    except Exception as e:
        return True, str(e).split("\n")[0]

    st, et = min([ e['s_sec'] for e in xml ]), max([ e['e_sec'] for e in xml ])
    base_time = edf.info['meas_date'] + timedelta(seconds=st)
    
    data = prep_psg_signal(edf.get_data(), transpose=True, fs=sfreq)
    # data = edf.get_data()

    X, Y = epoching_with_events(data, xml, sfreq=sfreq)
    n_missing_ch = dataset.n_missing_ch
    return {"x": X, "y": Y}

def load_only_edf(path_edf:str, epoch_start_time:str, sfreq:int=50, missing_ch='raise'):
    assert missing_ch in ['raise', 0, 1, 2], "Choose missing_ch in ['raise', 0, 1, 2]"
    try:
        edf, n_missing_ch = edf_io.load(
            path       = path_edf, 
            preload    = True, 
            resample   = sfreq, 
            preset     = "STAGENET", 
            exclude    = True,
            missing_ch = 'raise')
    except Exception as e:
        return True, str(e).split("\n")[0]
    
    base_time = edf.info['meas_date'].replace(tzinfo=None)

    if epoch_start_time == None:
        start_time = base_time
    else:
        start_time = datetime.strptime(epoch_start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=None)

    data = prep_psg_signal(edf.get_data(), transpose=True, fs=sfreq)
    X = epoching_from_time(data, base_time, start_time, sfreq=sfreq)

    return X


if __name__ == '__main__':
    edf_dir = "/home/honeynaps/data/250718_CND/EDF2"
    error_list = []
    fill_na = False
    sfreq = 50
    with open("errors.txt", "w") as f:
        filenames = [filename.replace(".edf", "") for filename in os.listdir(edf_dir)]
        for filename in filenames:
            has_error, message = main(filename, fill_na=fill_na, sfreq=sfreq)

            if has_error:
                line = f"{filename},{message}"
                print(line)
                error_list.append(line)
                f.write(line + "\n")

