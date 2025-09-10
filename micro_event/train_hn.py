import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from models.crop_models import REDv2Time, REDv2CWT1D
from models.REDv2TimePSD import REDv2TimePSD
from datasets.dataset_hn import SleepEventDatasetEBX
from datasets.dataset_hn_mc import SleepEventDatasetEBXMC
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support
from losses import masked_focal_loss, CustomASLLossBinary
from postprocess.postprocessor import upsample_preds, postprocess_preds_by_length
from common.seed import set_seed
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm


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
    model.eval()
    all_probs   = []   # positive-class probabilities
    all_labels  = []   # ground-truth labels (0/1)
    total_loss  = 0.0

    with torch.no_grad():
        for X, y, mask in val_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)

            # ── forward ───────────────────────────────────────────────
            logits = model(X)
            if logits.ndim > 2:
                logits = logits.squeeze(1)
            # logits = logits[:, 65:-65]                 # border 잘라내기

            if mask.ndim > 2:  mask = mask.squeeze(1)
            if y.ndim   > 2:   y    = y.squeeze(1)

            probs = torch.softmax(logits, dim=-1)[..., 1]  # P(y=1)

            # ── valid 위치만 수집 ─────────────────────────────────────
            valid_mask = mask.bool()
            all_probs.append(probs[valid_mask].cpu())
            all_labels.append(y[valid_mask].cpu())

            loss = masked_focal_loss(logits, y, mask)
            total_loss += loss.item() * X.size(0)

    # ── 하나의 1-D 텐서/배열로 병합 ──────────────────────────────────
    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # ── AUPRC ────────────────────────────────────────────────────────
    auprc = average_precision_score(all_labels, all_probs)

    # ── PR-커브 및 best threshold 탐색 ───────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    # precision, recall 길이는 thresholds보다 1만큼 큽니다 → 마지막 점 제외
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = f1s.argmax()

    best_thr       = thresholds[best_idx]
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])
    best_f1        = float(f1s[best_idx])
    loss           = total_loss / len(val_loader.dataset)  # 평균 손실

    all_probs = all_probs.reshape(-1) 
    # all_probs = postprocess_preds_by_length(all_probs, min_len_sec=0.15, max_sec=5, fs=25, threshold=best_thr)
    # p, r, f1, _ = precision_recall_fscore_support(all_labels, all_probs, average='binary', zero_division=0)
    # print(f"After postprocessing: "
    #       f"Precision = {p:.4f}, Recall = {r:.4f}, F1 = {f1:.4f}")

    return loss, auprc, best_thr, best_precision, best_recall, best_f1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='time')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--page_duration', type=int, default=20)  # seconds
    parser.add_argument('--event_type', type=str, default='kcomplex')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', type=str2bool, default=False)
    parser.add_argument('--test_page', type=str, default='N2')
    parser.add_argument('--expand', type=float, default=0.0)
    parser.add_argument('--border_sec', type=float, default=2.6)  # seconds
    parser.add_argument('--pretrained', type=str2bool, default=False)
    parser.add_argument('--mc', type=str2bool, default=False)
    parser.add_argument('--tag', type=str, default='',)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)


    data_dir = "/home/honeynaps/data/HN_DATA_MW"  
    subjects = os.listdir(data_dir + "/" + "EDF2")
    subjects = [s.split(".")[0] for s in subjects if s.endswith(".edf")]

    all_files = ['SCH-241031R1_M-30-OV-MO',
                 'SCH-241024R4_F-30-NW-NO',
                 'SCH-230114R3_M-60-OV-SE',
                 'SCH-190921R1_M-40-OV-SE',
                 'SCH-230106R2_F-40-OB-MI',
                 'SCH-180426R2_F-60-OV-MO']
    # invalid_ids = ['SCH-241031R1_M-30-OV-MO', 'SCH-241024R4_F-30-NW-NO']
    invalid_ids = []
    # val_ids = subjects[3:]  # Adjust as needed
    val_ids = ['SCH-190921R1_M-40-OV-SE', 'SCH-230114R3_M-60-OV-SE']
    train_ids = [s for s in subjects if s not in invalid_ids and s not in val_ids]

    print(f"Train IDs: {train_ids}")
    print(f"Validation IDs: {val_ids}")
    print(f"Invalid IDs: {invalid_ids}")

    dataset_cls = SleepEventDatasetEBXMC if args.mc else SleepEventDatasetEBX

    train_dataset = dataset_cls(data_dir, event_type=args.event_type,
                                subject_ids=train_ids,
                                border_sec=args.border_sec,
                                pages_subset="N2",
                                expand_sec=args.expand,
                                page_duration=args.page_duration,                                      
                                augmented_page=True)
    val_dataset  = dataset_cls(data_dir, page_duration=args.page_duration,
                               event_type=args.event_type,
                               border_sec=args.border_sec,
                               pages_subset=args.test_page,
                               subject_ids=val_ids)
    
    # is_pos = (train_dataset.marks.sum(axis=1) > 0)
    # pos_idx = np.where(is_pos)[0]
    # neg_idx = np.where(~is_pos)[0]

    # # to make ∑w_pos ≈ ∑w_neg, give each positive a larger weight
    # w_pos = 1.0 / len(pos_idx)
    # w_neg = 1.0 / len(neg_idx)
    # weights = torch.zeros(len(train_dataset))
    # weights[pos_idx] = w_pos
    # weights[neg_idx] = w_neg

    # EPOCH_SIZE = 2 * len(neg_idx)        # e.g. match #neg → ~50 % positives
    # sampler = WeightedRandomSampler(weights,
    #                                 num_samples=EPOCH_SIZE,
    #                                 replacement=True)
 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                            #    sampler=sampler,
                                               num_workers=4)
    val_loader  = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=4)

    save_dir = "/home/honeynaps/data/eis/SEED_pytorch/saved_models"

    # model = SleepEventDetector(input_channels=1)
    # model = SEEDModel(in_channels=1)
    in_channels = 1 if not args.mc else 4

    if args.model == 'time':
        print("Using REDv2Time model")
        # model = REDv2TimePSD(in_channels=in_channels)
        model = REDv2Time(in_channels=in_channels)
    else:
        print("Using REDv2CWT1D model")
        model = REDv2CWT1D()

    if args.event_type == 'kcomplex':
        pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/model_kcomplex_ep167_f10.9576_no_invalid.pth'
        pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/model_kcomplex_ep125_f10.9384_norm.pth'
    elif args.event_type == 'spindle':
        pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/model_spindle_ep166_f10.9372_no_invalid.pth'
        pretrained_path = '/home/honeynaps/data/eis/SEED_pytorch/saved_models/model_spindle_ep159_f10.9315_norm.pth'

    if args.pretrained and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))

    # criterion = CustomASLLossBinary().to(device)
    criterion = masked_focal_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)

    if args.pretrained:
        loss, auprc, best_thr, precision, recall, f1_score = evaluate_model(model, val_loader, device)
        print(f"Initial PT Score) Val Loss = {loss:.4f}, "
              f"Val AUPRC = {auprc:.4f}, Best Threshold = {best_thr:.4f}, "
              f"Val Precision = {precision:.4f}, Val Recall = {recall:.4f}, Val F1 = {f1_score:.4f}")
        

    num_epochs = 50
    best_valf1 = 0.43

    for epoch in range(1, num_epochs + 1):
        print(f"========Model:{args.model}|Pretrained:{args.pretrained}|EXP:{args.expand:.3f}", end='')
        print(f"{args.event_type}|{args.tag}|MC:{args.mc}|TEST:{args.test_page}=========")
    
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        for X, y, mask in tqdm(train_loader, dynamic_ncols=True, desc=f"Epoch {epoch:02d}"):
            # Move data to device
            X = X.to(device)
            y = y.to(device).float()      
            mask = mask.to(device).float()
            
            optimizer.zero_grad()
            logits = model(X)             

            if logits.ndim > 2:
                logits = logits.squeeze(1)

            # logits = logits[:, 65:-65]
            if y.ndim > 2:
                y = y.squeeze(1)
                mask = mask.squeeze(1)
            # Compute masked focal loss for this batch
            loss = criterion(logits, y, mask)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X.size(0)  # accumulate scaled by batch size

        avg_train_loss = total_train_loss / len(train_dataset)

        _, auprc, best_thr, precision, recall, f1_score = evaluate_model(model, train_loader, device)
        print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.4f}, "
              f"Train AUPRC = {auprc:.4f}, Best Threshold = {best_thr:.4f}, "
              f"Train Precision = {precision:.4f}, Train Recall = {recall:.4f}, Train F1 = {f1_score:.4f}")

        loss, auprc, best_thr, precision, recall, f1_score = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch:03d}: Val Loss = {loss:.4f}, "
              f"Val AUPRC = {auprc:.4f}, Best Threshold = {best_thr:.4f}, "
              f"Val Precision = {precision:.4f}, Val Recall = {recall:.4f}, Val F1 = {f1_score:.4f}")
        
        if f1_score > best_valf1:
            best_valf1 = f1_score
            print(f"Validation F1 score improved to {best_valf1:.4f} at epoch {epoch:03d}")
            if args.save:
                model_path = f"HN_{args.event_type}_ep{epoch:03d}_f1{best_valf1:.4f}_{args.tag}_th{best_thr:.4f}.pth"
                model_path = os.path.join(save_dir, model_path)
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path} with F1 score {best_valf1:.4f}")
        else:
            print(f"No improvement in F1 score. Current best: {best_valf1:.4f}")