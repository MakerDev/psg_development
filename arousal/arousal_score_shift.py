import os
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from utils.tools import load_edf_file

from ArousalFinalShift import ArousalFinalShift



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
#--DEF

def save_to_xml(preds, save_path, base_time, description, location):
    root = ET.Element("annotationlist")

    for pe in preds:
        onset_time = base_time + timedelta(seconds=pe[0])
        
        annotation = ET.SubElement(root, "annotation")
        onset_elem = ET.SubElement(annotation, "onset")
        onset_elem.text = onset_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        duration_elem = ET.SubElement(annotation, "duration")
        duration_elem.text = f"{pe[1]:.6f}"

        desc_elem = ET.SubElement(annotation, "description")
        desc_elem.text = description

        location_elem = ET.SubElement(annotation, "location")
        location_elem.text = location

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)

    tree.write(save_path, encoding="UTF-8", xml_declaration=True)
#--DEF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str)
    parser.add_argument('--dest', type=str)
    parser.add_argument('--start_time', type=str, default=None, help='Start time in format "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ver', type=int, default=2)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--type', type=str, default='union', choices=['time', 'spec', 'union', 'intersection'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not (args.edf or args.dest) :
        print('Arguments "--edf" or "--dest" required!!!')
        os._exit(1)
    #--IF

    edf = load_edf_file(
        path       = args.edf, 
        preload    = True, 
        resample   = args.fs, 
        preset     = "STAGENET", 
        exclude    = True,
        missing_ch = 'raise')
    
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
    
    detector = ArousalFinalShift(sigs, base_time, 
                 start_time  =start_time,
                 gpu         =args.gpu,
                 seed        =args.seed,
                 num_channels=args.num_channels,
                 fs          =args.fs,
                 type        =args.type,
                 ver         =args.ver,
                 tag         =args.tag)

    pretrained_dir = "/home/honeynaps/data/shared/arousal/saved_models"
    preds = detector(pretrained_dir)

    edf_name = os.path.basename(args.edf)
    xml_name = edf_name.replace('.edf', '_AROUS.xml')
    edf_path = args.edf
    save_path = os.path.join(args.dest, xml_name)
    save_to_xml(preds, save_path, base_time, 'AROUS-SPONT', edf.ch_names[0])
    print(f'Saved XML at: {save_path}')
#--MAIN