import torch
import os
import argparse
from datetime import datetime

from models.crop_models import REDv2Time
from datasets.dataset_hn_pred import SleepEventDatasetEBX
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from losses import masked_focal_loss, CustomASLLossBinary
from postprocess.postprocessor import evaluate_edf, merge_and_prune, postprocess_preds
from common.eval_utils import event_level_analysis
from util.tools import save_micro_events_by_channels, save_micro_events_by_channels_and_type

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--edf', type=str, default='/home/honeynaps/data/HN_DATA_MW/250718_VF/EDF/SCH-241218R1_M-60-OV-MI.edf')
    parser.add_argument('--dest', type=str, default='/home/honeynaps/data/shared/micro_event/preds')
    parser.add_argument('--start_time', type=str, default=None, help='Start time in format "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--page_duration', type=int, default=10)  # seconds
    parser.add_argument('--event_type', type=str, default='spindle', choices=['kcomplex', 'spindle'])
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

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
                                              batch_size=args.batch_size,
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

    if args.pretrained and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

    preds_all = evaluate_model(model, data_loader, device, th=th,
                               max_len_sec=max_len_sec, min_len_sec=min_len_sec)

    preds_all = postprocess_preds(preds_all, sleep_dataset,
                                  event_type=args.event_type,
                                  page_duration=args.page_duration)
    
    base_time = sleep_dataset.get_start_time()
    filename = os.path.basename(args.edf)
    filename = filename.replace('.edf', f'_{args.event_type.upper()}.xml')
    xml_save_path = os.path.join(args.dest, filename)

    desc = "MW_EEG-KCOMP" if args.event_type == 'kcomplex' else "MW_EEG-SPIND"
    
    save_micro_events_by_channels(base_time, preds_all, sfreq=200//8,
                                  xml_save_path=xml_save_path,
                                  description=desc,
                                  min_duration=0)
