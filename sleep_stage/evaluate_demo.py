import random
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import natsort
import argparse
import csv
from sklearn.metrics import confusion_matrix, classification_report
from models.cnn_encoders import *
from utils.transforms import *
from utils.post_process import run_postprocess

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, label_list, num_channels=9, fs=50, transforms=None):
        self.data_list = [torch.tensor(data[:, :, :num_channels], dtype=torch.float32) for data in data_list]
        self.label_list = [torch.tensor(labels, dtype=torch.long) for labels in label_list]
        self.num_data = [data.shape[0] for data in self.data_list]
        self.data_list = torch.concat(self.data_list, dim=0).unsqueeze(1)
        self.label_list = torch.concat(self.label_list, dim=0)
        self.num_channels = num_channels
        self.fs = fs
        self.transform = transforms

        self.data_list, self.label_list = self._group_data(self.data_list, self.label_list, 1)
        self._permute_data()

    def _group_data(self, data, labels, n):
        grouped_data = []
        grouped_labels = []
        for idx in range(0, len(data) - n + 1):
            grouped_data.append(data[idx:idx+n]) 
            grouped_labels.append(labels[idx+n-1])  # Label for the last item in the group
        
        grouped_data = torch.stack(grouped_data)
        grouped_labels = torch.tensor(grouped_labels, dtype=torch.long)
        
        return grouped_data, grouped_labels

    def from_where(self, idx):
        for i, num in enumerate(self.num_data):
            if idx < num:
                return i, idx
            idx -= num
        return -1, -1

    def get_from(self, user_idx, idx):
        final_idx = 0
        for i in range(user_idx):
            final_idx += self.num_data[i]
        final_idx += idx
        return self.__getitem__(final_idx)

    def _permute_data(self):     
        self.data_list = self.data_list.reshape(-1, 1, self.data_list.size(3), self.data_list.size(4))
        self.data_list = self.data_list.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]

        if self.transform:
            original_shape = data.shape
            data = data.reshape(self.num_channels, -1)
            data, label = self.transform(data, label)
            data = data.reshape(original_shape)

        return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--target', type=str, default='pred', choices=['pred', 'true'])
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    val_transforms = ["NormaliseOnly"] # Inference 시에는 Normalise만 사용합니다.
    val_transforms = build_transforms(val_transforms, n_channels=args.num_channels)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_type = 'dataset_mixed'

    edf_dir = f'/home/honeynaps/data/GOLDEN/EDF2'
    xml_dir = f'/home/honeynaps/data/GOLDEN/EBX2/SLEEP'

    dataset_dir = f'/home/honeynaps/data/GOLDEN/PICKLE/SLEEP_50_NOFILL'

    edf_dir = f'/home/honeynaps/data/250718_CND/EDF'
    xml_dir = f'/home/honeynaps/data/250718_CND/EBX/SLEEP'

    dataset_dir = f'/home/honeynaps/data/250718_CND/PICKLE/SLEEP_50_NOFILL_NOPREP'

    file_names = natsort.natsorted(os.listdir(dataset_dir))
    
    pretrained = False
    pin_memory = True
    num_channels = args.num_channels

    if args.model == 'resnet18':
        model = resnet18(num_channels=num_channels, pretrained=pretrained)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    save_path = '/home/honeynaps/data/shared/sleep_stage/saved_models/pretrained_asam_ver3.pt'
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model = model.to(device)
    model.eval()

    print("Total Users: ", len(file_names))

    save_dir = "/home/honeynaps/data/shared/sleep_stage"
    result_filename = f"eval_output{args.tag}.csv"

    acc_mean = 0
    y_true_all, y_pred_all = [], []
    with open(os.path.join(save_dir, result_filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["User Index", "Filename", "Used", "Accuracy", "Acc0", "Acc1", "Acc2", "Acc3", "Acc4", "NS0", "NS1", "NS2", "NS3", "NS4"])

        cm_integrated = None
        for file_idx, file_name in enumerate(file_names):
            with open(os.path.join(dataset_dir, file_name), 'rb') as f:
                data_dict = pickle.load(f)

            val_dataset = CustomDataset([data_dict['x']], [data_dict['y']], args.num_channels, args.fs, transforms=val_transforms)
            test_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=args.batch_size,
                pin_memory=True, 
                shuffle=False
            )

            acc = 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    data = data.reshape(-1, args.num_channels, 1, 1500 * (args.fs // 50))
                    outputs = model(data)
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    labels = labels.reshape(-1)
                    acc += (outputs.argmax(dim=-1) == labels).float().mean().item()

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(outputs.argmax(dim=-1).cpu().numpy())

            # y_pred = run_postprocess(y_pred, 6)
            acc = np.mean(np.array(y_pred) == np.array(y_true))
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            acc_mean += acc
            cm = confusion_matrix(y_true, y_pred)

            report = classification_report(y_true, y_pred, zero_division=0)
            labels_wise_acc = cm.diagonal() / cm.sum(axis=1)
            labels_wise_acc = np.nan_to_num(labels_wise_acc, nan=0).tolist()

            filename = file_name.split('.')[0]
            used = "val"
            num_samples_per_label = cm.sum(axis=1).tolist()

            if len(num_samples_per_label) < 5:
                num_samples_per_label.extend([0] * (5 - len(labels_wise_acc)))

            if len(labels_wise_acc) < 5:
                labels_wise_acc.extend([0] * (5 - len(labels_wise_acc)))

            writer.writerow([file_idx, filename, used, acc] + labels_wise_acc + num_samples_per_label)

            print(f"#User{file_idx} - Results for file: {filename} - ({used})")
            print("Accuracy:", acc)
            print("Label-wise accuracy:", labels_wise_acc)
            print("Num samples per label:", num_samples_per_label)
            print("Confusion Matrix:\n", cm)
            print("-" * 50)

    cm_integrated = confusion_matrix(y_true_all, y_pred_all)
    print("Integrated Confusion Matrix:\n", cm_integrated)
    print("Mean Accuracy: ", acc_mean / len(file_names))