import argparse
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.losses import CustomBCEWithLogitsLoss, CustomAsymmetricLoss, BoundaryAwareAsymmetricLoss
from utils.score2018 import Challenge2018ScoreVer2
from models.DeepSleepFinal import DeepSleepFinal


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CombinedFeatureDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        spec = torch.from_numpy(data["spectrogram"]).float()
        time_features = torch.from_numpy(data["time_features"]).float()
        labels = torch.from_numpy(data["y"]).float()

        record_id = os.path.splitext(os.path.basename(file_path))[0]
        inputs = {"spectrogram": spec, "time_features": time_features}
        return inputs, labels, record_id


def combined_collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, str]]):
    max_time = max(item[0]["spectrogram"].shape[-1] for item in batch)
    batch_size = len(batch)

    spec_channels = batch[0][0]["spectrogram"].shape[0]
    spec_freqs = batch[0][0]["spectrogram"].shape[1]
    time_channels = batch[0][0]["time_features"].shape[0]
    time_features = batch[0][0]["time_features"].shape[1]

    spec_batch = torch.zeros(batch_size, spec_channels, spec_freqs, max_time)
    time_batch = torch.zeros(batch_size, time_channels, time_features, max_time)
    label_batch = torch.full((batch_size, max_time), -1.0)
    record_ids = []

    for idx, (inputs, labels, record_id) in enumerate(batch):
        length = inputs["spectrogram"].shape[-1]
        spec_batch[idx, :, :, :length] = inputs["spectrogram"]
        time_batch[idx, :, :, :length] = inputs["time_features"]
        label_batch[idx, :length] = labels
        record_ids.append(record_id)

    inputs = {"spectrogram": spec_batch, "time_features": time_batch}
    return inputs, label_batch, record_ids


def eval_fn(model, loader, device):
    model.eval()
    criterion = CustomBCEWithLogitsLoss().to(device)
    scores = Challenge2018ScoreVer2()

    loss_epoch_sum = 0.0
    val_auroc = 0.0
    val_auprc = 0.0
    val_best_f1 = 0.0
    val_best_thr = 0.0

    with torch.no_grad():
        for inputs, targets, record_ids in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            loss_epoch_sum += float(loss.item())

            probs = torch.sigmoid(logits).cpu()
            targets_cpu = targets.cpu()

            for i, record_id in enumerate(record_ids):
                y_target = targets_cpu[i]
                y_pred = probs[i]
                mask = y_target > -0.5
                y_target = y_target[mask]
                y_pred = y_pred[mask]
                if len(y_target) == 0:
                    continue
                scores.score_record(y_target.numpy(), y_pred.numpy(), record_id)
                val_auroc += scores.record_auroc(record_id)
                val_auprc += scores.record_auprc(record_id)
                val_best_f1 += scores.record_f1(record_id)
                val_best_thr += scores.record_best_threshold(record_id)

    num_records = len(loader.dataset)
    if num_records > 0:
        val_auroc /= num_records
        val_auprc /= num_records
        val_best_f1 /= num_records
        val_best_thr /= num_records

    mean_loss = loss_epoch_sum / max(len(loader), 1)
    return mean_loss, val_auroc, val_auprc, val_best_f1, val_best_thr


def build_model(sample_batch: Dict[str, torch.Tensor]) -> DeepSleepFinal:
    spec_channels = sample_batch["spectrogram"].shape[1]
    time_feature_channels = sample_batch["time_features"].shape[1] * sample_batch["time_features"].shape[2]
    model = DeepSleepFinal(
        n_spectrogram_channels=spec_channels,
        n_time_feature_channels=time_feature_channels,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train DeepSleepFinal with combined features")
    parser.add_argument("--data_dir", type=str, default="./data/combined", help="Directory with combined pickle files")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--loss", type=str, choices=["bce", "asl", "ba_asl"], default="bce")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    file_paths = [
        os.path.join(args.data_dir, f)
        for f in sorted(os.listdir(args.data_dir))
        if f.endswith(".pkl")
    ]

    if len(file_paths) == 0:
        raise RuntimeError(f"No pickle files found in {args.data_dir}. Run prep_spectrogram_combined.py first.")

    random.shuffle(file_paths)
    split_idx = int(len(file_paths) * (1.0 - args.val_ratio))
    train_files = file_paths[:split_idx]
    val_files = file_paths[split_idx:]

    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError("Not enough data to create train/validation splits. Adjust --val_ratio or add more data.")

    train_dataset = CombinedFeatureDataset(train_files)
    val_dataset = CombinedFeatureDataset(val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=combined_collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=combined_collate_fn,
        drop_last=False,
    )

    sample_inputs, _, _ = next(iter(train_loader))
    model = build_model(sample_inputs)
    model = model.to(args.device)

    if args.loss == "bce":
        criterion = CustomBCEWithLogitsLoss().to(args.device)
    elif args.loss == "asl":
        criterion = CustomAsymmetricLoss().to(args.device)
    else:
        criterion = BoundaryAwareAsymmetricLoss().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_auprc = 0.0
    best_epoch = -1
    patience_counter = 0
    best_checkpoint_path = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets, _ in train_loader:
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            targets = targets.to(args.device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item())

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_auroc, val_auprc, val_f1, val_thr = eval_fn(model, val_loader, args.device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f} | Val F1: {val_f1:.4f} | Val Th: {val_thr:.3f}")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch
            patience_counter = 0
            best_checkpoint_path = os.path.join(args.output_dir, f"deepsleep_final_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_auprc": val_auprc,
                "val_auroc": val_auroc,
                "val_f1": val_f1,
                "val_thr": val_thr,
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation AUPRC: {best_val_auprc:.4f} at epoch {best_epoch}")
    if best_checkpoint_path is not None:
        print(f"Best model saved to {best_checkpoint_path}")


if __name__ == "__main__":
    main()
