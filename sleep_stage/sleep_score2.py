import os
import argparse
from datetime import datetime

from models.cnn_encoders import *
from modules.iofiles import edf as edf_io
from utils.transforms import *
from utils.tools import *
from prep_window_wise import load_edf_for_demo, load_only_edf
from utils.post_process import run_postprocess

from SleepFinal import SleepFinal


def save_to_xml(edf_path, y, save_path, base_time, fs=50, probs=None):
    if base_time is None:
        raw = load_edf_file(edf_path, preload=True, resample=fs, preset="STAGENET", exclude=True, missing_ch='raise')
        base_time = raw.info['meas_date']
    else:
        base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
    save_sleepstage_xml(base_time, y, save_path, probs=probs)
#--DEF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default="/home/honeynaps/data/dataset/EDF/SCH-180210R2_M-40-OV-MI.edf")
    parser.add_argument('--dest', type=str, default="/home/honeynaps/data/shared/sleep_stage")
    parser.add_argument('--start_time', type=str, default=None)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--nofill', type=str2bool, default=True)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    if not (args.edf or args.dest) :
        print('Arguments "--edf" or "--dest" required!!!')
        os._exit(1)
    #--IF

    edf, n_missing_ch = edf_io.load(
        path       = args.edf, 
        preload    = True, 
        resample   = args.fs, 
        preset     = "STAGENET", 
        exclude    = True,
        missing_ch = 0)
    
    base_time = edf.info['meas_date'].replace(tzinfo=None)
    if args.start_time == None:
        start_time = base_time
    else:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=None)
    #--IF
    
    SID_MAP = { 
        'F3-':'F3_2', 'F4-':'F4_1', 'C3-':'C3_2', 'C4-':'C4_1', 'O1-':'O1_2', 'O2-':'O2_1', 
        'LOC':'LOC' , 'ROC':'ROC', 
        'EMG':'CHIN'
    }
    data = edf.get_data()

    sigs = {}
    for i in range(len(edf.ch_names)) :
        name = edf.ch_names[i]
        if name in SID_MAP :
            sigs[SID_MAP[name]] = data[i]
        else :
            sigs[SID_MAP[name[:3]]] = data[i]
        #--IF
    #--FOR
    del sigs['C4_1']

    detector = SleepFinal(sigs, base_time, 
                 start_time  =start_time,
                 model       =args.model,
                 gpu         =args.gpu,
                 seed        =args.seed,
                 num_channels=args.num_channels,
                 fs          =args.fs,
                 nofill      =args.nofill,
                 tag         =args.tag)

    pretrained_dir = f'/home/honeynaps/data/shared/sleep_stage/saved_models'
    y_pred, all_probs = detector(pretrained_dir, n_missing_ch)

    save_dir = args.dest
    edf_name = os.path.basename(args.edf)
    save_path = os.path.join(save_dir, edf_name.replace('.edf', '_SLEEP.xml'))
    save_to_xml(args.edf, y_pred, save_path, base_time=args.start_time, fs=args.fs, probs=all_probs)
    print(f'Saved {save_path}')
#--MAIN