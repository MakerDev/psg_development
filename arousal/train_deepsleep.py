import os
import numpy as np
import torch
import pickle
import random
import datetime
import torch.utils.tensorboard as tb
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from utils.losses import *
from utils.score2018 import Challenge2018ScoreVer2
from models.DeepSleepFinal import DeepSleepMultimodal


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==============================
# Dataset for Multimodal Chunks
# ==============================
class MultimodalArousalDataset(Dataset):
    """
    Dataset for multimodal arousal detection with 60-second chunks
    Each pickle file contains:
        - x_time_raw: (time, channels)
        - x_time_combined: (channels, 4, time)
        - x_spec: (channels, freq, time_bins)
        - stat_features: (channels, 6, time_windows)
        - y_time: (time,)
        - y_spec: (time_bins,)
    """
    def __init__(self, file_paths, normalize=True, use_time_label=True):
        super().__init__()
        self.file_paths = file_paths
        self.normalize = normalize
        self.use_time_label = use_time_label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        with open(path, 'rb') as f:
            data_dict = pickle.load(f)

        # Load features
        x_time_combined = data_dict['x_time_combined']  # (channels, 4, time)
        x_spec = data_dict['x_spec']                    # (channels, freq, time_bins)
        x_stat = data_dict['stat_features']             # (channels, 6, time_windows)

        # Load labels
        if self.use_time_label:
            y = data_dict['y_time']  # (time,)
        else:
            y = data_dict['y_spec']  # (time_bins,)

        # Convert to torch tensors
        x_time_combined = torch.from_numpy(x_time_combined).float()
        x_spec = torch.from_numpy(x_spec).float()
        x_stat = torch.from_numpy(x_stat).float()
        y = torch.from_numpy(y).float()

        return x_time_combined, x_spec, x_stat, y, idx


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads to the maximum length in the batch
    """
    x_time_list, x_spec_list, x_stat_list, y_list, idx_list = zip(*batch)

    # Find max lengths
    max_time = max([x.shape[2] for x in x_time_list])
    max_spec_time = max([x.shape[2] for x in x_spec_list])
    max_stat_time = max([x.shape[2] for x in x_stat_list])
    max_y = max([len(y) for y in y_list])

    # Pad time features
    x_time_batch = []
    for x in x_time_list:
        if x.shape[2] < max_time:
            pad = torch.zeros(x.shape[0], x.shape[1], max_time - x.shape[2])
            x = torch.cat([x, pad], dim=2)
        x_time_batch.append(x)
    x_time_batch = torch.stack(x_time_batch, dim=0)

    # Pad spec features
    x_spec_batch = []
    for x in x_spec_list:
        if x.shape[2] < max_spec_time:
            pad = torch.zeros(x.shape[0], x.shape[1], max_spec_time - x.shape[2])
            x = torch.cat([x, pad], dim=2)
        x_spec_batch.append(x)
    x_spec_batch = torch.stack(x_spec_batch, dim=0)

    # Pad stat features
    x_stat_batch = []
    for x in x_stat_list:
        if x.shape[2] < max_stat_time:
            pad = torch.zeros(x.shape[0], x.shape[1], max_stat_time - x.shape[2])
            x = torch.cat([x, pad], dim=2)
        x_stat_batch.append(x)
    x_stat_batch = torch.stack(x_stat_batch, dim=0)

    # Pad labels (use -1 for padding)
    y_batch = []
    for y in y_list:
        if len(y) < max_y:
            pad = torch.ones(max_y - len(y)) * -1
            y = torch.cat([y, pad], dim=0)
        y_batch.append(y)
    y_batch = torch.stack(y_batch, dim=0)

    idx_tensor = torch.LongTensor(idx_list)

    return x_time_batch, x_spec_batch, x_stat_batch, y_batch, idx_tensor


# ==============================
# Evaluation Function
# ==============================
def eval_fn(model, loader, device, criterion, comp_score=True):
    model.eval()

    scores = Challenge2018ScoreVer2() if comp_score else None

    with torch.no_grad():
        loss_epoch_sum = 0
        val_auroc, val_auprc, best_f1, best_th = 0, 0, 0, 0

        for x_time, x_spec, x_stat, y, idx in tqdm(loader, desc="Evaluating"):
            x_time = x_time.to(device)
            x_spec = x_spec.to(device)
            x_stat = x_stat.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x_time, x_spec, x_stat, comp=comp_score)

            # Resize y_pred to match y
            if y_pred.shape[2] != y.shape[1]:
                y_pred = torch.nn.functional.interpolate(
                    y_pred, size=y.shape[1], mode='linear', align_corners=False
                )

            y_pred = y_pred.squeeze(1)  # (B, T)

            # Compute loss
            loss = criterion(y_pred, y)

            # Compute AUROC/AUPRC score for each record in batch
            if comp_score:
                for i, single_idx in enumerate(idx):
                    record_name = str(single_idx.item())
                    y_target = y[i].view(-1).cpu()
                    y_pred_i = y_pred[i].view(-1).cpu()
                    y_pad_mask = y_target != -1

                    if y_pad_mask.sum() == 0:
                        continue

                    scores.score_record(y_target[y_pad_mask], y_pred_i[y_pad_mask], record_name)
                    auroc = scores.record_auroc(record_name)
                    auprc = scores.record_auprc(record_name)
                    f1 = scores.record_f1(record_name)
                    th = scores.record_best_threshold(record_name)

                    val_auroc += auroc
                    val_auprc += auprc
                    best_f1 += f1
                    best_th += th

            loss_epoch_sum += float(loss.item())

        if comp_score and len(loader.dataset) > 0:
            val_auroc /= len(loader.dataset)
            val_auprc /= len(loader.dataset)
            best_f1 /= len(loader.dataset)
            best_th /= len(loader.dataset)

            print(f"Best F1: {best_f1:.4f}, Best Threshold: {best_th:.4f}")

    return loss_epoch_sum / len(loader), val_auroc, val_auprc, best_th


# ==============================
# Main Training Script
# ==============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                       default='/home/honeynaps/data/250718_CND/AROUS_MULTIMODAL/AROUSAL_COMBINED_50_multimodal_60s',
                       help='Directory containing preprocessed multimodal chunks')
    parser.add_argument('--model', type=str, default='DeepSleepMultimodal')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--base_ch', type=int, default=32, help='Base number of channels in model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='asl', choices=['bce', 'asl', 'ba_asl'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--use_tb', type=str2bool, default=False)
    parser.add_argument('--use_attention', type=str2bool, default=True, help='Use attention fusion')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data files
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please run prep_spectrogram_combined.py first to generate the data.")
        exit(1)

    file_list = [f for f in os.listdir(args.data_dir) if f.endswith('.pkl')]
    file_paths = [os.path.join(args.data_dir, f) for f in file_list]

    if len(file_paths) == 0:
        print(f"Error: No pickle files found in {args.data_dir}")
        print("Please run prep_spectrogram_combined.py first to generate the data.")
        exit(1)

    print(f"Found {len(file_paths)} chunk files")

    # Split train/val
    random.shuffle(file_paths)
    train_ratio = 0.8
    split_index = int(train_ratio * len(file_paths))
    train_files = file_paths[:split_index]
    val_files = file_paths[split_index:]

    print(f"Training chunks: {len(train_files)}")
    print(f"Validation chunks: {len(val_files)}")

    # Create datasets
    train_dataset = MultimodalArousalDataset(train_files, normalize=True, use_time_label=True)
    val_dataset = MultimodalArousalDataset(val_files, normalize=True, use_time_label=True)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Setup experiment name
    current_setting = (f"{args.model}_CH{args.num_channels}_BCH{args.base_ch}_"
                      f"BS{args.batch_size}_LR{args.lr}_{args.loss}_"
                      f"ATT{args.use_attention}_{args.tag}")

    tb_name = f'{str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{current_setting}'
    if args.use_tb:
        TB_WRITER = tb.SummaryWriter(f'./tensorboards/{tb_name}')

    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    model = DeepSleepMultimodal(
        n_channels=args.num_channels,
        base_ch=args.base_ch,
        use_attention=args.use_attention
    ).to(device)

    print(f"Model created: {args.model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Setup loss
    if args.loss == 'bce':
        criterion = CustomBCELoss().to(device)
    elif args.loss == 'asl':
        criterion = CustomAsymmetricLoss().to(device)
    elif args.loss == 'ba_asl':
        criterion = BoundaryAwareAsymmetricLoss().to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Training loop
    best_train_auprc, best_val_auprc = 0, 0
    best_model_path = None

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}: {current_setting}")
        model.train()

        train_loss_sum = 0
        for i, (x_time, x_spec, x_stat, y, idx) in enumerate(tqdm(train_dataloader, desc="Training")):
            optimizer.zero_grad()

            x_time = x_time.to(device)
            x_spec = x_spec.to(device)
            x_stat = x_stat.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x_time, x_spec, x_stat, comp=False)

            # Resize y_pred to match y
            if y_pred.shape[2] != y.shape[1]:
                y_pred = torch.nn.functional.interpolate(
                    y_pred, size=y.shape[1], mode='linear', align_corners=False
                )

            y_pred = y_pred.squeeze(1)  # (B, T)

            # Compute loss
            loss = criterion(y_pred, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        # Evaluation
        train_loss, train_auroc, train_auprc, train_th = eval_fn(
            model, train_dataloader, device, criterion, comp_score=True
        )
        print(f"Train - Loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f}")

        if train_auprc > best_train_auprc:
            best_train_auprc = train_auprc

        val_loss, val_auroc, val_auprc, val_th = eval_fn(
            model, val_dataloader, device, criterion, comp_score=True
        )
        print(f"Val - Loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")

        print(f"Best Train AUPRC: {best_train_auprc:.4f}, Best Val AUPRC: {best_val_auprc:.4f}")
        print("=" * 80)

        # Save best model
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc

            # Create save directory if needed
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)

            # Remove previous best model
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)

            # Save new best model
            best_model_path = os.path.join(
                args.save_dir,
                f"deepsleep_multimodal_ep{epoch:03d}_auprc{val_auprc:.4f}_th{val_th:.4f}.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")

        # TensorBoard logging
        if args.use_tb:
            TB_WRITER.add_scalar('Loss/train', train_loss, epoch)
            TB_WRITER.add_scalar('Loss/val', val_loss, epoch)
            TB_WRITER.add_scalar('AUROC/train', train_auroc, epoch)
            TB_WRITER.add_scalar('AUROC/val', val_auroc, epoch)
            TB_WRITER.add_scalar('AUPRC/train', train_auprc, epoch)
            TB_WRITER.add_scalar('AUPRC/val', val_auprc, epoch)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation AUPRC: {best_val_auprc:.4f}")
    print(f"Best model saved at: {best_model_path}")

    if args.use_tb:
        TB_WRITER.close()
