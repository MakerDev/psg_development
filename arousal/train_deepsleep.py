"""
Training script for multimodal arousal detection

Trains DeepSleepFinal model on combined time-domain and frequency-domain features
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import datetime
import torch.utils.tensorboard as tb
from torch.utils.data import Dataset, DataLoader

# Import model
from models.DeepSleepFinal import DeepSleepFinal

# Import utilities
from utils.losses import *
from utils.score2018 import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==================== DATASET ====================

class MultimodalArousalDataset(Dataset):
    """
    Dataset for multimodal arousal detection

    Loads preprocessed pickle files containing:
        - x_time: (C, 6, T) time-domain features
        - x_spec: (C, F, T_spec) frequency-domain features
        - y_time: (T,) time-domain labels
        - y_spec: (T_spec,) spectrogram labels
    """
    def __init__(self, file_paths, max_time_len=2**21, add_noise=True, noise_level=0.02,
                 spec_downsample=4):
        super().__init__()
        self.file_paths = file_paths
        self.max_time_len = max_time_len
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.spec_downsample = spec_downsample  # Downsample factor for spectrogram

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        with open(path, 'rb') as f:
            data = pickle.load(f)

        x_time = data['x_time']  # (C, 6, T)
        x_spec = data['x_spec']  # (C, F, T_spec)
        y_time = data['y_time']  # (T,)

        # Reshape time features: (C, 6, T) -> (C*6, T)
        C, n_features, T = x_time.shape
        x_time = x_time.reshape(C * n_features, T)

        # Pad or crop to max_time_len
        x_time, y_time = self._pad_or_crop(x_time, y_time, self.max_time_len)

        # Downsample spectrogram to reduce memory usage
        # This reduces temporal resolution but keeps frequency info
        if self.spec_downsample > 1:
            x_spec = self._downsample_spec(x_spec, self.spec_downsample)

        # Pad or crop spectrogram to match time length
        # Spectrogram time bins should be roughly max_time_len / (hop_size * downsample)
        # With nperseg=50, noverlap=25, hop_size=25, time_bins ≈ T/25
        max_spec_time = self.max_time_len // (25 * self.spec_downsample)
        x_spec = self._pad_or_crop_spec(x_spec, max_spec_time)

        # Add noise for augmentation (only during training)
        if self.add_noise:
            noise = np.random.normal(0, self.noise_level, x_time.shape).astype(np.float32)
            x_time = x_time + noise

        # Convert to tensors
        x_time = torch.from_numpy(x_time).float()
        x_spec = torch.from_numpy(x_spec).float()
        y_time = torch.from_numpy(y_time).float()

        return x_time, x_spec, y_time, idx

    def _pad_or_crop(self, x, y, target_len):
        """Pad or crop to target length"""
        current_len = x.shape[1]

        if current_len < target_len:
            # Pad
            pad_len = target_len - current_len
            x = np.pad(x, ((0, 0), (0, pad_len)), mode='constant')
            y = np.pad(y, (0, pad_len), mode='constant', constant_values=-1)
        elif current_len > target_len:
            # Crop
            x = x[:, :target_len]
            y = y[:target_len]

        return x, y

    def _downsample_spec(self, x_spec, factor):
        """
        Downsample spectrogram along time axis to reduce memory
        Args:
            x_spec: (C, F, T_spec) spectrogram
            factor: downsampling factor
        Returns:
            x_spec_ds: (C, F, T_spec//factor) downsampled spectrogram
        """
        if factor <= 1:
            return x_spec

        C, F, T = x_spec.shape
        new_T = T // factor

        # Reshape and average over factor
        # (C, F, T) -> (C, F, new_T, factor) -> (C, F, new_T)
        x_spec_reshaped = x_spec[:, :, :new_T*factor].reshape(C, F, new_T, factor)
        x_spec_ds = x_spec_reshaped.mean(axis=3)

        return x_spec_ds

    def _pad_or_crop_spec(self, x_spec, target_time_bins):
        """
        Pad or crop spectrogram to target time bins
        Args:
            x_spec: (C, F, T_spec) spectrogram
            target_time_bins: target number of time bins
        Returns:
            x_spec: (C, F, target_time_bins)
        """
        current_time_bins = x_spec.shape[2]

        if current_time_bins < target_time_bins:
            # Pad
            pad_len = target_time_bins - current_time_bins
            x_spec = np.pad(x_spec, ((0, 0), (0, 0), (0, pad_len)), mode='constant')
        elif current_time_bins > target_time_bins:
            # Crop from center
            start = (current_time_bins - target_time_bins) // 2
            x_spec = x_spec[:, :, start:start+target_time_bins]

        return x_spec


# ==================== EVALUATION ====================

def eval_fn(model, loader, device, criterion, comp_score=True):
    """Evaluate model on validation set"""
    model.eval()

    scores = Challenge2018ScoreVer2() if comp_score else None

    with torch.no_grad():
        loss_epoch_sum = 0
        val_auroc, val_auprc, best_f1, best_th = 0, 0, 0, 0

        for x_time, x_spec, y, idx in loader:
            x_time = x_time.to(device)
            x_spec = x_spec.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x_time, x_spec, apply_sigmoid=comp_score)

            # Compute loss
            loss = criterion(y_pred.squeeze(1), y)

            # Compute scores
            if comp_score:
                for i, single_idx in enumerate(idx):
                    record_name = str(single_idx.item())
                    y_target = y[i].view(-1).cpu()
                    y_pred_i = y_pred[i].view(-1).cpu()
                    y_pad_mask = y_target != -1

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

        val_auroc /= len(loader.dataset)
        val_auprc /= len(loader.dataset)
        best_f1 /= len(loader.dataset)
        best_th /= len(loader.dataset)

        print(f"Best F1: {best_f1:.4f}, Best Threshold: {best_th:.4f}")

    return loss_epoch_sum/len(loader), val_auroc, val_auprc, best_th


# ==================== MAIN TRAINING ====================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train multimodal arousal detection model')

    # Model parameters
    parser.add_argument('--n_channels', type=int, default=9, help='Number of EEG channels')
    parser.add_argument('--n_time_features', type=int, default=6, help='Number of time-domain features per channel')
    parser.add_argument('--time_base_ch', type=int, default=8,
                        help='Base channels for time branch (reduced from 16 to save memory)')
    parser.add_argument('--freq_base_ch', type=int, default=8,
                        help='Base channels for frequency branch (reduced from 16 to save memory)')
    parser.add_argument('--freq_layers', type=int, default=3,
                        help='Number of layers in frequency branch (reduced from 4 to save memory)')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    parser.add_argument('--chunk_size', type=int, default=2**17,
                        help='Chunk size for processing (default 2^17, reduces memory significantly)')
    parser.add_argument('--chunk_overlap', type=float, default=0.25,
                        help='Overlap ratio between chunks (default 0.25)')

    # Training parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--loss', type=str, default='asl', choices=['bce', 'asl', 'ba_asl'], help='Loss function')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Data parameters
    parser.add_argument('--data_dir', type=str,
                        default='/home/honeynaps/data/250718_CND/AROUS_MULTIMODAL/AROUSAL_MULTIMODAL_50_multimodal_v1',
                        help='Directory containing preprocessed multimodal data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--max_time_len', type=int, default=2**19,
                        help='Maximum time length (default 2^19 = ~10s at 50Hz, reduces memory)')
    parser.add_argument('--spec_downsample', type=int, default=4,
                        help='Spectrogram downsampling factor to reduce memory (default 4)')
    parser.add_argument('--add_noise', type=str2bool, default=True, help='Add noise augmentation')
    parser.add_argument('--noise_level', type=float, default=0.02, help='Noise level for augmentation')

    # Logging
    parser.add_argument('--use_tb', type=str2bool, default=False, help='Use tensorboard')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save models')
    parser.add_argument('--tag', type=str, default='', help='Experiment tag')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load data files
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist!")
        print(f"Please run prep_spectrogram_combined.py first to generate multimodal features.")
        exit(1)

    file_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.pkl')]

    if len(file_paths) == 0:
        print(f"Error: No pickle files found in {args.data_dir}")
        exit(1)

    print(f"Found {len(file_paths)} files in {args.data_dir}")

    # Split train/val
    random.shuffle(file_paths)
    split_idx = int(args.train_ratio * len(file_paths))
    train_files = file_paths[:split_idx]
    val_files = file_paths[split_idx:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create datasets
    train_dataset = MultimodalArousalDataset(
        train_files,
        max_time_len=args.max_time_len,
        add_noise=args.add_noise,
        noise_level=args.noise_level,
        spec_downsample=args.spec_downsample
    )

    val_dataset = MultimodalArousalDataset(
        val_files,
        max_time_len=args.max_time_len,
        add_noise=False,  # No noise for validation
        noise_level=0.0,
        spec_downsample=args.spec_downsample
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DeepSleepFinal(
        n_channels=args.n_channels,
        n_time_features=args.n_time_features,
        time_base_ch=args.time_base_ch,
        freq_base_ch=args.freq_base_ch,
        freq_layers=args.freq_layers,
        dropout=args.dropout,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    if args.loss == 'bce':
        criterion = CustomBCEWithLogitsLoss().to(device)
    elif args.loss == 'asl':
        criterion = CustomAsymmetricLoss().to(device)
    elif args.loss == 'ba_asl':
        criterion = BoundaryAwareAsymmetricLoss().to(device)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Tensorboard
    current_setting = f"MultiModal_TimeCh{args.time_base_ch}_FreqCh{args.freq_base_ch}_" + \
                      f"BS{args.batch_size}_{args.loss}_LR{args.lr}_Drop{args.dropout}"
    if args.tag:
        current_setting += f"_{args.tag}"

    tb_name = f'{str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{current_setting}'

    if args.use_tb:
        tb_dir = './tensorboards'
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        TB_WRITER = tb.SummaryWriter(f'{tb_dir}/{tb_name}')

    # Training loop
    best_train_auprc, best_val_auprc = 0, 0
    best_model_path = None

    print(f"\nStarting training: {current_setting}\n")

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}: {current_setting}")
        model.train()

        epoch_loss = 0
        for i, (x_time, x_spec, y, idx) in enumerate(train_loader):
            optimizer.zero_grad()

            x_time = x_time.to(device)
            x_spec = x_spec.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x_time, x_spec, apply_sigmoid=False)

            # Compute loss
            loss = criterion(y_pred.squeeze(1), y)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Evaluate on train set
        train_loss, train_auroc, train_auprc, train_th = eval_fn(
            model, train_loader, device, criterion, comp_score=True
        )
        print(f"Train loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f}")

        if train_auprc > best_train_auprc:
            best_train_auprc = train_auprc

        # Evaluate on validation set
        val_loss, val_auroc, val_auprc, val_th = eval_fn(
            model, val_loader, device, criterion, comp_score=True
        )
        print(f"Validation loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")

        # Update learning rate
        scheduler.step(val_auprc)

        # Save best model
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc

            # Remove previous best model
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)

            # Save new best model
            best_model_path = os.path.join(
                args.save_dir,
                f"deepsleep_multimodal_auprc{val_auprc:.4f}_th{val_th:.4f}.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")

        print(f"Best Train AUPRC: {best_train_auprc:.4f} | Best Val AUPRC: {best_val_auprc:.4f}")
        print("=" * 80)

        # Tensorboard logging
        if args.use_tb:
            TB_WRITER.add_scalar('Loss/train', train_loss, epoch)
            TB_WRITER.add_scalar('Loss/val', val_loss, epoch)
            TB_WRITER.add_scalar('AUROC/train', train_auroc, epoch)
            TB_WRITER.add_scalar('AUROC/val', val_auroc, epoch)
            TB_WRITER.add_scalar('AUPRC/train', train_auprc, epoch)
            TB_WRITER.add_scalar('AUPRC/val', val_auprc, epoch)
            TB_WRITER.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

    print(f"\nTraining completed!")
    print(f"Best validation AUPRC: {best_val_auprc:.4f}")
    print(f"Best model saved at: {best_model_path}")

    if args.use_tb:
        TB_WRITER.close()
