import os
import sys
import argparse
import xml.etree.ElementTree as ET
import torch
from datetime import datetime, timedelta

from arousal.utils.tools import load_edf_file
from arousal.ArousalFinal import ArousalFinal
from sleep_stage.SleepFinal import SleepFinal
from sleep_stage.modules.iofiles import edf as edf_io
from sleep_stage.utils.post_process import run_postprocess
from micro_event.models.crop_models import REDv2Time
from micro_event.datasets.dataset_hn_pred import SleepEventDatasetEBX
from micro_event.util.tools import save_micro_events_by_channels, save_micro_events_by_channels_and_type
from micro_event.postprocess.postprocessor import evaluate_edf, merge_and_prune, postprocess_preds
from tools.post_process_enhanced import correct_sleep_stages_with_tracking, PostProcessInfo
from tools.utils import str2bool, load_sleep_stage
from sklearn.metrics import confusion_matrix


def save_enhanced_sleepstage_xml(meas_date, original_stages, corrected_stages, 
                                post_process_info, ground_truth, xml_save_path, 
                                location="EEG-F4"):
    """
    Save sleep stage predictions with detailed post-processing analysis to XML.
    
    Format: {STAGE}_{TRUE/FALSE}_{POST_PROCESS_WRONG}_{POST_PROCESS_REASON}
    - STAGE: Sleep stage (SLEEP-W, SLEEP-R, SLEEP-1, SLEEP-2, SLEEP-3)
    - TRUE/FALSE: Whether the prediction matches ground truth
    - POST_PROCESS_WRONG: Whether post-processing caused a correct prediction to become wrong
    - POST_PROCESS_REASON: What post-processing rule caused the change
    """
    label_to_stage = {
        0: "SLEEP-W",
        1: "SLEEP-R", 
        2: "SLEEP-1",
        3: "SLEEP-2",
        4: "SLEEP-3"
    }

    root = ET.Element("annotationlist")

    for i, (original_stage, corrected_stage, gt_stage) in enumerate(zip(original_stages, corrected_stages, ground_truth)):
        start_sec = i * 30
        onset_time = meas_date + timedelta(seconds=start_sec)
        duration = 30.0
        
        # Get post-processing info for this epoch
        pp_info = post_process_info[i] if i < len(post_process_info) else None
        
        # Build description string
        stage_name = label_to_stage.get(corrected_stage, "SLEEP-U")
        
        # Check if prediction is correct
        is_correct = "TRUE" if corrected_stage == gt_stage else "FALSE"
        
        # Check if post-processing caused error
        post_process_wrong = "FALSE"
        post_process_reason = "NONE"
        
        if pp_info and pp_info.was_changed:
            # Original was correct but corrected is wrong
            if original_stage == gt_stage and corrected_stage != gt_stage:
                post_process_wrong = "TRUE"
            post_process_reason = pp_info.change_reason
        
        # Format: {STAGE}_{TRUE/FALSE}_{POST_PROCESS_WRONG}_{POST_PROCESS_REASON}
        description = f"{stage_name}_{is_correct}_{post_process_wrong}_{post_process_reason}"

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

    # XML tree 작성
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)

    # XML 파일로 저장
    tree.write(xml_save_path, encoding="UTF-8", xml_declaration=True)


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


def evaluate_model(model, val_loader, device, th=0.1557, max_len_sec=5, min_len_sec=0.5):
    model.eval()
    all_probs = {}

    with torch.no_grad():
        for X, info in val_loader:        
            X = X.to(device)

            logits = model(X)                       
            if logits.ndim > 2:
                logits = logits.squeeze(1)          

            probs = torch.softmax(logits, dim=-1)[..., 1]

            batch_size = X.size(0)
            channel_names = info[1] if isinstance(info, (list, tuple)) else ['default'] * batch_size

            for b in range(batch_size):
                ch_name = channel_names[b]
                all_probs.setdefault(ch_name, []).append(probs[b].cpu())


    for ch_name in all_probs:
        all_probs[ch_name]  = torch.cat(all_probs[ch_name],  dim=0).numpy()  # (ΣT,)

    all_preds = {}
    for ch_name in all_probs:
        all_probs[ch_name]  = all_probs[ch_name].reshape(-1)
        all_preds[ch_name]  = (all_probs[ch_name] > th).astype(int)
        all_preds[ch_name]  = merge_and_prune(all_preds[ch_name], fs=200//8, 
                                              max_len_sec=max_len_sec,
                                              min_len_sec=min_len_sec,
                                              merge_th=0.1)
    

    return all_preds


def pred_arousal(args):
    edf = load_edf_file(
        path       = args.edf, 
        preload    = True, 
        resample   = 50, 
        preset     = "STAGENET", 
        exclude    = True,
        missing_ch = 'raise')
    
    base_time = edf.info['meas_date'].replace(tzinfo=None)
    if args.start_time == None:
        start_time = base_time
    else:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=None)
    
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
    
    detector = ArousalFinal(sigs, base_time, 
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

    return preds, base_time

def pred_sleep_stage(args):
    if not (args.edf or args.dest) :
        print('Arguments "--edf" or "--dest" required!!!')
        os._exit(1)

    edf, n_missing_ch = edf_io.load(
        path       = args.edf, 
        preload    = True, 
        resample   = 50, 
        preset     = "STAGENET", 
        exclude    = True,
        missing_ch = 'raise')
    
    base_time = edf.info['meas_date'].replace(tzinfo=None)
    if args.start_time == None:
        start_time = base_time
    else:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=None)
    
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

    detector = SleepFinal(sigs, base_time, 
                 start_time  =start_time,
                 model       ='resnet18',
                 gpu         =args.gpu,
                 seed        =42,
                 num_channels=9,
                 fs          =50,
                 nofill      =True,
                 tag         =args.tag)

    pretrained_dir = f'/home/honeynaps/data/shared/sleep_stage/saved_models'
    y_pred, all_probs = detector(pretrained_dir)

    return y_pred


def pred_micro_event(args, th_mul=1.3):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    file_names = [args.edf]

    if args.start_time:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=None)
    else:
        start_time = None

    sleep_dataset = SleepEventDatasetEBX(
                                file_names,
                                start_times=[start_time],
                                page_duration=args.page_duration)
    data_loader = torch.utils.data.DataLoader(sleep_dataset,
                                              batch_size=32,
                                              shuffle=False, num_workers=4)

    save_dir = "/home/honeynaps/data/shared/micro_event/saved_models"

    model = REDv2Time(in_channels=1)
    model.to(device)

    if args.event_type == 'kcomplex':
        pretrained_path = f'{save_dir}/HN_kcomplex_ep012_f10.4473_newall_th0.2433.pth'
        th, max_len_sec, min_len_sec = 0.1557, 5, 0.5
    elif args.event_type == 'spindle':
        pretrained_path = f'{save_dir}/HN_spindle_ep006_f10.5243_newall_th0.2657.pth'
        th, max_len_sec, min_len_sec = 0.2725, 3, 0.5
    th = th * th_mul
    if args.pretrained and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

    preds_all = evaluate_model(model, data_loader, device, th=th,
                               max_len_sec=max_len_sec, min_len_sec=min_len_sec)

    preds_all = postprocess_preds(preds_all, sleep_dataset,
                                  event_type=args.event_type,
                                  page_duration=args.page_duration)
    
    return preds_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default="/home/honeynaps/data/250718_CND/EDF2/CND-241021R1_M-60-NW-SE.edf")
    parser.add_argument('--dest', type=str, default="/home/honeynaps/data/shared/integrate")
    parser.add_argument('--start_time', type=str, default=None, help='Start time in format "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ver', type=int, default=2)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--type', type=str, default='spec', choices=['time', 'spec', 'union', 'intersection'])
    parser.add_argument('--seed', type=int, default=0)

    # Micro Event Parameters
    parser.add_argument('--page_duration', type=int, default=10)  # seconds
    parser.add_argument('--event_type', type=str, default='kcomplex', choices=['kcomplex', 'spindle'])
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--th_mul', type=float, default=1.3, help='Threshold multiplier for micro event detection')

    args = parser.parse_args()

    if not (args.edf or args.dest):
        print('Arguments "--edf" or "--dest" required!!!')
        os._exit(1)

    # Extract EDF filename without extension
    edf_basename = os.path.splitext(os.path.basename(args.edf))[0]
    
    # Load ground truth labels
    sleep_label_xml_path = args.edf.replace('.edf', '_SLEEP.xml').replace('EDF2', 'EBX/SLEEP').replace('EDF', 'EBX/SLEEP')
    sleep_labels, start_time = load_sleep_stage(sleep_label_xml_path)

    args.start_time = start_time

    # Run predictions
    print("Running arousal prediction...")
    arousal_preds, base_time = pred_arousal(args)
    
    print("Running sleep stage prediction...")
    sleep_preds = pred_sleep_stage(args)
    original_sleep_preds = sleep_preds.copy()
    # sleep_preds = run_postprocess(sleep_preds, 6)
    
    print("Running micro event prediction...")
    micro_event_preds_by_channels = pred_micro_event(args)
                                  
    # Align lengths
    min_length = min(len(sleep_preds), len(sleep_labels))
    sleep_preds = sleep_preds[:min_length]
    original_sleep_preds = original_sleep_preds[:min_length]
    sleep_labels = sleep_labels[:min_length]
                                                                                                                                                     
    print("Sleep Accuracy before post-processing correction:")
    n_corrected, n_total = 0, 0                                          
    for i in range(len(sleep_preds)):
        if sleep_preds[i] == sleep_labels[i]:
            n_corrected += 1
        n_total += 1
    print(f"Corrected: {n_corrected}, Total: {n_total}, Accuracy: {n_corrected / n_total:.2f}")
    cm = confusion_matrix(sleep_labels, sleep_preds)
    print("Confusion Matrix:")
    print(cm)

    # Apply post-processing correction with tracking
    print("Applying post-processing correction...")
    corrected_sleep_stages, post_process_info = correct_sleep_stages_with_tracking(
        arousal_preds, 
        sleep_preds, 
        micro_event_preds_by_channels
    )

    print("Sleep Accuracy after post-processing correction:")
    n_corrected, n_total = 0, 0
    for i in range(len(corrected_sleep_stages)):
        if corrected_sleep_stages[i] == sleep_labels[i]:
            n_corrected += 1
        n_total += 1
    print(f"Corrected: {n_corrected}, Total: {n_total}, Accuracy: {n_corrected / n_total:.2f}")
    cm = confusion_matrix(sleep_labels, corrected_sleep_stages)
    print("Confusion Matrix:")
    print(cm)

    # Analyze post-processing impact
    changes_count = sum(1 for info in post_process_info if info.was_changed)
    correct_to_wrong = sum(1 for i, info in enumerate(post_process_info) 
                          if info.was_changed and sleep_preds[i] == sleep_labels[i] and corrected_sleep_stages[i] != sleep_labels[i])
    wrong_to_correct = sum(1 for i, info in enumerate(post_process_info) 
                          if info.was_changed and sleep_preds[i] != sleep_labels[i] and corrected_sleep_stages[i] == sleep_labels[i])
    
    print(f"\nPost-processing Analysis:")
    print(f"Total epochs changed: {changes_count}")
    print(f"Correct -> Wrong: {correct_to_wrong}")
    print(f"Wrong -> Correct: {wrong_to_correct}")
    print(f"Net improvement: {wrong_to_correct - correct_to_wrong}")

    # Save enhanced XML with post-processing analysis
    xml_filename = f"{edf_basename}_SLEEP_ANALYSIS.xml"
    xml_save_path = os.path.join(args.dest, xml_filename)
    
    print(f"Saving enhanced XML to: {xml_save_path}")
    save_enhanced_sleepstage_xml(
        meas_date=base_time,
        original_stages=sleep_preds,
        corrected_stages=corrected_sleep_stages,
        post_process_info=post_process_info,
        ground_truth=sleep_labels,
        xml_save_path=xml_save_path
    )

    print(f"EDF: {args.edf}")
    print(f"Event Type: {args.event_type}")
    print(f"TH_MUL: {args.th_mul}")
    print(f"XML saved to: {xml_save_path}")
    print("Analysis complete!")