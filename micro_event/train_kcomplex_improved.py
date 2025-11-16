"""
Improved K-Complex Detection Training Script

This script trains the KComplexDetector model with shape-aware loss functions.

Key improvements:
1. Multi-task learning (detection + peak location + zero-crossing)
2. Shape-aware loss functions
3. Zero-crossing boundary refinement
4. Amplitude and duration constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import argparse
from tqdm import tqdm

from models.kcomplex_detector import KComplexDetector
from losses_kcomplex import KComplexLoss
from datasets.dataset_hn import SleepEventDatasetEBX
from datasets.dataset_hn_mc import SleepEventDatasetEBXMC
from sklearn.metrics import precision_recall_curve, average_precision_score
from postprocess.kcomplex_postprocessor import postprocess_kcomplex_predictions


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def evaluate_model(model, val_loader, criterion, device, use_auxiliary=True):
    """
    Evaluate model on validation set

    Args:
        model: KComplexDetector
        val_loader: DataLoader
        criterion: KComplexLoss
        device: torch device
        use_auxiliary: whether to use auxiliary tasks

    Returns:
        loss, auprc, best_threshold, precision, recall, f1_score, loss_dict
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    loss_components = {
        'detection': 0.0,
        'peak_align': 0.0,
        'peak_order': 0.0,
        'zerocross': 0.0,
        'shape': 0.0
    }

    with torch.no_grad():
        for X, y, mask in val_loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)

            # Forward pass
            if use_auxiliary:
                outputs = model(X, return_auxiliary=True)
                logits = outputs['logits']
            else:
                logits = model(X, return_auxiliary=False)
                outputs = logits

            # Ensure correct shapes
            if logits.ndim > 2:
                logits = logits.squeeze(1) if logits.shape[1] == 1 else logits
            if mask.ndim > 2:
                mask = mask.squeeze(1)
            if y.ndim > 2:
                y = y.squeeze(1)

            # Get probabilities
            probs = torch.softmax(logits, dim=-1)[..., 1]  # P(y=1)

            # Collect valid positions
            valid_mask = mask.bool()
            all_probs.append(probs[valid_mask].cpu())
            all_labels.append(y[valid_mask].cpu())

            # Compute loss
            loss, loss_dict = criterion(outputs, y, mask, raw_signal=X)
            total_loss += loss.item() * X.size(0)

            # Accumulate loss components
            for key in loss_components.keys():
                if key in loss_dict:
                    loss_components[key] += loss_dict[key] * X.size(0)

    # Concatenate all predictions
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute metrics
    auprc = average_precision_score(all_labels, all_probs)

    # Find best threshold
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = f1s.argmax()

    best_thr = thresholds[best_idx]
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])
    best_f1 = float(f1s[best_idx])

    avg_loss = total_loss / len(val_loader.dataset)

    # Average loss components
    for key in loss_components.keys():
        loss_components[key] /= len(val_loader.dataset)

    return avg_loss, auprc, best_thr, best_precision, best_recall, best_f1, loss_components


def main():
    parser = argparse.ArgumentParser(description='Train improved K-complex detector')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--page_duration', type=int, default=20, help='Page duration in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--save', type=str2bool, default=True, help='Save best model')
    parser.add_argument('--test_page', type=str, default='N2', help='Test page type')
    parser.add_argument('--border_sec', type=float, default=2.6, help='Border duration in seconds')
    parser.add_argument('--use_auxiliary', type=str2bool, default=True, help='Use auxiliary tasks')
    parser.add_argument('--tag', type=str, default='improved', help='Experiment tag')

    # Loss weights (UPDATED for strict K-complex detection)
    parser.add_argument('--weight_detection', type=float, default=1.0, help='Detection loss weight')
    parser.add_argument('--weight_peak_align', type=float, default=0.4, help='Peak alignment loss weight (was 0.3)')
    parser.add_argument('--weight_peak_order', type=float, default=0.3, help='Peak ordering loss weight (was 0.2)')
    parser.add_argument('--weight_zerocross', type=float, default=0.2, help='Zero-crossing loss weight')
    parser.add_argument('--weight_shape', type=float, default=0.5, help='Shape consistency loss weight (was 0.1 - NOW CRITICAL)')

    # Amplitude thresholds (CLINICAL STANDARDS)
    parser.add_argument('--min_amplitude', type=float, default=75, help='Minimum K-complex amplitude in µV (clinical standard)')
    parser.add_argument('--max_amplitude', type=float, default=300, help='Maximum K-complex amplitude in µV')

    args = parser.parse_args()

    # Set random seeds
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 80)
    print("K-Complex Detection Training - Improved Architecture")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Use auxiliary tasks: {args.use_auxiliary}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Border duration: {args.border_sec}s")
    print(f"Page duration: {args.page_duration}s")
    print("=" * 80)

    # Data paths
    data_dir = "/home/honeynaps/data/HN_DATA_MW"
    save_dir = "/home/honeynaps/data/eis/SEED_pytorch/saved_models"

    # Subject IDs
    subjects = os.listdir(data_dir + "/" + "EDF2")
    subjects = [s.split(".")[0] for s in subjects if s.endswith(".edf")]

    invalid_ids = []
    val_ids = ['SCH-190921R1_M-40-OV-SE', 'SCH-230114R3_M-60-OV-SE']
    train_ids = [s for s in subjects if s not in invalid_ids and s not in val_ids]

    print(f"Train subjects: {len(train_ids)}")
    print(f"Validation subjects: {len(val_ids)}")
    print()

    # Create datasets
    dataset_cls = SleepEventDatasetEBX

    train_dataset = dataset_cls(
        data_dir,
        event_type='kcomplex',
        subject_ids=train_ids,
        border_sec=args.border_sec,
        pages_subset="N2",
        expand_sec=0.0,
        page_duration=args.page_duration,
        augmented_page=True
    )

    val_dataset = dataset_cls(
        data_dir,
        page_duration=args.page_duration,
        event_type='kcomplex',
        border_sec=args.border_sec,
        pages_subset=args.test_page,
        subject_ids=val_ids
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Create model
    model = KComplexDetector(in_channels=1)
    model.to(device)

    print("Model architecture:")
    print(f"  Input channels: 3 (raw + abs + derivative)")
    print(f"  Multi-task outputs: detection + peaks + zero-crossing")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create loss function
    criterion = KComplexLoss(
        weight_detection=args.weight_detection,
        weight_peak_align=args.weight_peak_align if args.use_auxiliary else 0.0,
        weight_peak_order=args.weight_peak_order if args.use_auxiliary else 0.0,
        weight_zerocross=args.weight_zerocross if args.use_auxiliary else 0.0,
        weight_shape=args.weight_shape,
        fs=200,
        min_amplitude=args.min_amplitude,  # STRICT clinical standard
        max_amplitude=args.max_amplitude
    ).to(device)

    print("Loss function configuration:")
    print("  Weights:")
    print(f"    Detection: {args.weight_detection}")
    if args.use_auxiliary:
        print(f"    Peak alignment: {args.weight_peak_align}")
        print(f"    Peak ordering: {args.weight_peak_order}")
        print(f"    Zero-crossing: {args.weight_zerocross}")
    print(f"    Shape consistency: {args.weight_shape} (CRITICAL)")
    print("  Amplitude constraints:")
    print(f"    Minimum: {args.min_amplitude} µV (clinical standard)")
    print(f"    Maximum: {args.max_amplitude} µV")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:03d}/{args.epochs:03d}")
        print("-" * 80)

        # Training phase
        model.train()
        total_train_loss = 0.0
        train_loss_components = {
            'detection': 0.0,
            'peak_align': 0.0,
            'peak_order': 0.0,
            'zerocross': 0.0,
            'shape': 0.0
        }

        pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)
        for X, y, mask in pbar:
            X = X.to(device)
            y = y.to(device).float()
            mask = mask.to(device).float()

            optimizer.zero_grad()

            # Forward pass
            if args.use_auxiliary:
                outputs = model(X, return_auxiliary=True)
            else:
                outputs = model(X, return_auxiliary=False)

            # Ensure correct shapes
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            if logits.ndim > 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            if y.ndim > 2:
                y = y.squeeze(1)
                mask = mask.squeeze(1)

            # Compute loss
            loss, loss_dict = criterion(outputs, y, mask, raw_signal=X)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X.size(0)

            # Accumulate loss components
            for key in train_loss_components.keys():
                if key in loss_dict:
                    train_loss_components[key] += loss_dict[key] * X.size(0)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Average training loss
        avg_train_loss = total_train_loss / len(train_dataset)
        for key in train_loss_components.keys():
            train_loss_components[key] /= len(train_dataset)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"  Detection: {train_loss_components['detection']:.4f}", end='')
        if args.use_auxiliary:
            print(f" | Peak align: {train_loss_components['peak_align']:.4f}", end='')
            print(f" | Peak order: {train_loss_components['peak_order']:.4f}", end='')
            print(f" | Zero-cross: {train_loss_components['zerocross']:.4f}", end='')
        print(f" | Shape: {train_loss_components['shape']:.4f}")

        # Validation phase
        val_loss, auprc, best_thr, precision, recall, f1_score, val_loss_components = \
            evaluate_model(model, val_loader, criterion, device, args.use_auxiliary)

        print(f"Val Loss: {val_loss:.4f} | AUPRC: {auprc:.4f} | Threshold: {best_thr:.4f}")
        print(f"Val Metrics: P={precision:.4f} R={recall:.4f} F1={f1_score:.4f}")

        # Save best model
        if f1_score > best_val_f1:
            best_val_f1 = f1_score
            best_epoch = epoch

            if args.save:
                model_filename = f"KC_improved_{args.tag}_ep{epoch:03d}_f1{f1_score:.4f}_th{best_thr:.4f}.pth"
                model_path = os.path.join(save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                print(f"✓ Model saved: {model_filename}")
        else:
            print(f"  (Best F1: {best_val_f1:.4f} at epoch {best_epoch})")

        print()

    print("=" * 80)
    print("Training completed!")
    print(f"Best F1 score: {best_val_f1:.4f} at epoch {best_epoch}")
    print("=" * 80)


if __name__ == "__main__":
    main()
