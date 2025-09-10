import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from models.crop_models import REDv2Time
from datasets.dataset_hn_eval import SleepEventDatasetEBX
from datasets.dataset_hn_mc import SleepEventDatasetEBXMC
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from losses import masked_focal_loss, CustomASLLossBinary
from postprocess.postprocessor import evaluate_edf, merge_and_prune
from common.eval_utils import event_level_analysis
from common.seed import set_seed
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


def evaluate_model(model, val_loader, device):
    """
    Batch-size가 1보다 클 때도 동작하도록 수정한 평가 루틴.
    ───────────────────────────────────────────────────────────
    반환값은 기존과 동일:
        loss, auprc, best_thr, best_precision, best_recall, best_f1
    """
    model.eval()
    all_probs, all_labels, all_masks = {}, {}, {}
    total_loss = 0.0

    with torch.no_grad():
        for X, y, mask, info in val_loader:         # (B, …)
            X, y, mask = X.to(device), y.to(device), mask.to(device)

            # ── forward ────────────────────────────────────────────
            logits = model(X)                       # (B, *, 2)
            if logits.ndim > 2:
                logits = logits.squeeze(1)          # (B, T, 2) → (B, T, 2) 그대로면 OK
            if mask.ndim > 2:
                mask = mask.squeeze(1)
            if y.ndim > 2:
                y = y.squeeze(1)

            probs = torch.softmax(logits, dim=-1)[..., 1]  # (B, T)

            # ── per-sample gather (채널별) ─────────────────────────
            batch_size = X.size(0)
            # info 구조: (파일명 list, 채널명 list, …) 라는 가정을 사용
            channel_names = info[1] if isinstance(info, (list, tuple)) else ['default'] * batch_size

            for b in range(batch_size):
                ch_name = channel_names[b]
                # 텐서는 detach() 뒤 CPU, numpy 변환은 나중에 한꺼번에
                all_probs.setdefault(ch_name, []).append(probs[b].cpu())
                all_labels.setdefault(ch_name, []).append(y[b].cpu())
                all_masks.setdefault(ch_name, []).append(mask[b].cpu())

            # ── 배치 단위 손실 ─────────────────────────────────────
            loss = masked_focal_loss(logits, y, mask)
            total_loss += loss.item() * batch_size

    for ch_name in all_probs:
        all_probs[ch_name]  = torch.cat(all_probs[ch_name],  dim=0).numpy()  # (ΣT,)
        all_labels[ch_name] = torch.cat(all_labels[ch_name], dim=0).numpy()
        all_masks[ch_name]  = torch.cat(all_masks[ch_name],  dim=0).numpy()

    # (채널 구분 없이 전체 지표)
    flat_probs  = np.concatenate(list(all_probs.values()),  axis=0)
    flat_labels = np.concatenate(list(all_labels.values()), axis=0)

    auprc = average_precision_score(flat_labels, flat_probs)

    precisions, recalls, thresholds = precision_recall_curve(flat_labels, flat_probs)
    f1s        = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx   = f1s.argmax()

    best_thr       = thresholds[best_idx] * 0.5
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])
    best_f1        = float(f1s[best_idx])
    mean_loss      = total_loss / len(val_loader.dataset)

    all_preds = {}
    for ch_name in all_probs:
        all_labels[ch_name] = all_labels[ch_name].reshape(-1)
        all_probs[ch_name]  = all_probs[ch_name].reshape(-1)
        all_preds[ch_name]  = (all_probs[ch_name] > best_thr).astype(int)
        all_preds[ch_name]  = merge_and_prune(all_preds[ch_name], fs=200//8, 
                                              max_len_sec=3,
                                              min_len_sec=0.5,
                                              merge_th=0.1)
    
    for ch_name in all_preds:
        mask = all_masks[ch_name].reshape(-1)
        preds = all_preds[ch_name]
        labels = all_labels[ch_name]

        total_preds = preds.sum()
        preds_in_mask = (preds * mask).sum()
        total_labels = labels.sum()
        labels_in_mask = (labels * mask).sum()
        print(f"Channel: {ch_name}, "
              f"Preds: {preds_in_mask}/{total_preds} - {preds_in_mask/total_preds if total_preds > 0 else 0:.4f}  "
              f"Labels: {labels_in_mask}/{total_labels} - {labels_in_mask/total_labels if total_labels > 0 else 0:.4f}")
    return mean_loss, auprc, best_thr, best_precision, best_recall, best_f1, (all_labels, all_preds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--page_duration', type=int, default=10)  # seconds
    parser.add_argument('--event_type', type=str, default='kcomplex', choices=['kcomplex', 'spindle'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--mc', type=str2bool, default=False)
    parser.add_argument('--tag', type=str, default='correct')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    data_dir = "/home/honeynaps/data/HN_DATA_MW"
    subjects = os.listdir(data_dir + "/" + "EDF2")
    subjects = [s.split(".")[0] for s in subjects if s.endswith(".edf")]

    file_names = ['SCH-190921R1_M-40-OV-SE'] # 'SCH-190921R1_M-40-OV-SE'
    file_names = ['SCH-230114R3_M-60-OV-SE'] # 'SCH-190921R1_M-40-OV-SE'

    dataset_cls = SleepEventDatasetEBXMC if args.mc else SleepEventDatasetEBX

    sleep_dataset = dataset_cls(data_dir, event_type=args.event_type,
                                subject_ids=file_names,
                                pages_subset="all",
                                page_duration=args.page_duration,                                      
                                augmented_page=False)
    data_loader = torch.utils.data.DataLoader(sleep_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=4)

    save_dir = "/home/honeynaps/data/eis/SEED_pytorch/saved_models"

    in_channels = 1 if not args.mc else 6
    model = REDv2Time(in_channels=in_channels)
    model.to(device)

    if args.event_type == 'kcomplex':
        # pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/HN_kcomplex_ep016_f10.4390_all_ch_th0.2575.pth'
        pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/HN_kcomplex_ep014_f10.4786_dur10_newdata_th0.2320.pth'
        pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/HN_kcomplex_ep012_f10.4473_newall_th0.2433.pth'
    elif args.event_type == 'spindle':
        # pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/HN_spindle_ep013_f10.4903_ch_std_th0.3164.pth'
        pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/HN_spindle_ep005_f10.5034_dur10_newdata_new_remap_th0.2853.pth'
        pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/HN_spindle_ep006_f10.5243_newall_th0.2657.pth'

    if args.pretrained and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

    loss, auprc, best_thr, precision, recall, f1_score, results = evaluate_model(model, data_loader, device)
    print(f"Initial PT Score) Val Loss = {loss:.4f}, "
            f"Val AUPRC = {auprc:.4f}, Best Threshold = {best_thr:.4f}, "
            f"Val Precision = {precision:.4f}, Val Recall = {recall:.4f}, Val F1 = {f1_score:.4f}")
    labels_all, preds_all = results

    matched_only = []
    labels_integrated = []
    preds_integrated  = []
    matched_only_by_channel = {}
    missed_only_by_channel  = {}
    wrong_only_by_channel   = {}
    labels_by_channel       = {}

    for channel_name in preds_all:
        preds = preds_all[channel_name].reshape(-1)
        labels = labels_all[channel_name].reshape(-1)
        stats = event_level_analysis(preds, labels, overlap_th=0.1, sfreq=200//8, return_stats=True)

        print(f"====Channel: {channel_name}====")
        for stat_key in stats:
            if 'only' in stat_key:
                continue
            print(f"{stat_key.capitalize()}: {stats[stat_key]:.4f}\t", end="")
        print()

        matched_only.extend(stats['matched_only'].tolist())
        labels_integrated.extend(labels.tolist())
        preds_integrated.extend(preds.tolist())
        matched_only_by_channel[channel_name] = stats['matched_only'].tolist()
        missed_only_by_channel[channel_name]  = stats['missed_only'].tolist()
        wrong_only_by_channel[channel_name]   = stats['wrong_only'].tolist()
        labels_by_channel[channel_name]       = labels.tolist()

    precisions, recalls, thresholds = precision_recall_curve(labels_integrated, preds_integrated)
    f1s        = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx   = f1s.argmax()

    best_thr       = thresholds[best_idx]
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])
    best_f1        = float(f1s[best_idx])
    print(f"Overall: Precision = {best_precision:.4f}, Recall = {best_recall:.4f}, F1 = {best_f1:.4f}")
    
    _, base_time = data_loader.dataset.get_start_time(0)
    save_path    = f'/home/honeynaps/data/eis/SEED_pytorch/preds/{file_names[0]}_{args.event_type.upper()}_{args.tag}.xml'
    # save_micro_events_by_channels(base_time, preds_all, sfreq=200//8,
    #                               xml_save_path=save_path, description="KCOMP", min_duration=0)
    # evaluate_edf(model, data_loader, threshold=best_thr, sfreq=200, save_path=save_path, device=device)

    preds_by_types = {
        "MATCHED": matched_only_by_channel,
        "MISSED" : missed_only_by_channel,
        "WRONG"  : wrong_only_by_channel, # 둘이 바뀌어있음. event_level_analysis에서 버그 있는 듯
    }
    save_micro_events_by_channels_and_type(base_time, preds_by_types, sfreq=200//8,
                                           xml_save_path=save_path, 
                                           description=args.event_type.upper()[:5], 
                                           min_duration=0)
