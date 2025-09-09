import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import natsort
import argparse
import random
import datetime
import torch.utils.tensorboard as tb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from models.cnn_encoders import *
from utils.transforms import *
from utils.losses import ASLSingleLabel

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, label_list, num_channels=9, fs=50, transforms=None):
        self.data_list = [torch.tensor(data[:, :, :num_channels], dtype=torch.float32) for data in data_list]
        self.label_list = [torch.tensor(labels, dtype=torch.long) for labels in label_list]
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
    parser.add_argument('--use_tb', type=str2bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--loss_weight', type=str, default='dynamic', choices=['dynamic','default'])
    parser.add_argument('--fs', type=int, default=50)
    parser.add_argument('--nofill', type=str2bool, default=True)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    transforms = ["NormaliseAndAddRandNoise"]
    val_transforms = ["NormaliseAndAddRandNoise"]
    tf_str = "_".join(transforms)
    transforms = build_transforms(transforms, n_channels=args.num_channels)
    val_transforms = build_transforms(val_transforms, n_channels=args.num_channels)

    use_tb = args.use_tb
    tb_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_WINDOW_WISE_SR{args.split_ratio}_{args.model}_{tf_str}_LR{args.lr}_BS{args.batch_size}_NC{args.num_channels}_{args.loss_weight}_FS{args.fs}'
    if args.nofill:
        tb_name += '_NOFILL'
    if len(args.tag) > 0:
        tb_name += '_' + args.tag

    if use_tb:
        TB_WRITER = tb.SummaryWriter(f'/home/honeynaps/data/tensorboards/{tb_name}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_dir = f'/home/honeynaps/data/dataset/PICKLE/SLEEP_{args.fs}'
    if args.nofill:
        dataset_dir += '_NOFILL'

    train_data_list = []
    train_label_list = []
    val_data_list = []
    val_label_list = []

    file_names = natsort.natsorted(os.listdir(dataset_dir))
    train_files = [file_name for file_name in file_names if 'CND' not in file_name]
    # file_name = file_names[:60]
    random_indices = np.random.permutation(len(train_files))
    train_files = [train_files[i] for i in random_indices]
    val_dir = f'/home/honeynaps/data/250718_CND/PICKLE/SLEEP_50_NOFILL_NOPREP'
    val_files = [file_name for file_name in file_names if 'CND' in file_name]

    # train_files = file_names[:int(args.split_ratio*len(file_names))]
    # val_files = file_names[int(args.split_ratio*len(file_names)):]

    for file_name in train_files:
        if file_name.endswith('.pickle'):
            with open(os.path.join(dataset_dir, file_name), 'rb') as f:
                data_dict = pickle.load(f)
                train_data_list.append(data_dict['x'])
                train_label_list.append(data_dict['y'])

    for file_name in val_files:
        if file_name.endswith('.pickle'):
            with open(os.path.join(dataset_dir, file_name), 'rb') as f:
                data_dict = pickle.load(f)
                val_data_list.append(data_dict['x'])
                val_label_list.append(data_dict['y'])

    pretrained = True
    pin_memory = True
    num_channels = args.num_channels
    train_dataset = CustomDataset(train_data_list, train_label_list, num_channels, args.fs, transforms=transforms)
    val_dataset = CustomDataset(val_data_list, val_label_list, num_channels, args.fs, transforms=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        pin_memory=pin_memory, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        pin_memory=pin_memory, 
        shuffle=False
    )

    if args.model == 'resnet18':
        model = resnet18(num_channels=num_channels, pretrained=pretrained)
    elif args.model == 'resnet50':
        model = resnet50(num_channels=num_channels, pretrained=pretrained)
    elif args.model == 'regnet128':
        model = regnet128(num_channels=num_channels, pretrained=pretrained)
    elif args.model == 'regnet16':
        model = regnet16(num_channels=num_channels, pretrained=pretrained)
    elif args.model == 'swin':
        model = swin_transformer(num_channels=num_channels)
    elif args.model == 'convnext':
        model = conv_next(num_channels=num_channels)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    num_signals = 1500 * (args.fs // 50)

    if args.loss_weight == 'default':
        class_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        vals_dict = {}
        for i in train_dataset.label_list.tolist():
            if i in vals_dict:
                vals_dict[i] += 1
            else:
                vals_dict[i] = 1
        total = sum(vals_dict.values())
        weight_dict = {k: (1 - (v/total)) for k, v in vals_dict.items()}
        class_weights = [weight_dict[i] for i in range(5)]

    criterion = ASLSingleLabel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001, amsgrad=True)

    num_epochs = 200
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(tb_name)
        model.train()
        running_loss = 0.0
        acc_mean = 0.0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            data = data.reshape(-1, num_channels, 1, num_signals)
            outputs = model(data)
            outputs = outputs.reshape(-1, outputs.size(-1))
            labels = labels.reshape(-1)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_mean += (outputs.argmax(dim=-1) == labels).float().mean().item()
            running_loss += loss.item()

        y_true, y_pred = [], []
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_acc_mean = 0.0
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                data = data.reshape(-1, num_channels, 1, num_signals)
                
                outputs = model(data)
                outputs = outputs.reshape(-1, outputs.size(-1))
                labels = labels.reshape(-1)

                loss = criterion(outputs, labels)
                val_acc_mean += (outputs.argmax(dim=-1) == labels).float().mean().item()
                val_loss += loss.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=-1).cpu().numpy())

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {acc_mean/len(train_loader):.4f}, Val Acc: {val_acc_mean/len(test_loader):.4f}')
        
        if val_acc_mean/len(test_loader) > best_val_acc:
            acc = val_acc_mean/len(test_loader)
            best_val_acc = val_acc_mean/len(test_loader)
            # Save model
            torch.save(model.state_dict(), f'/home/honeynaps/data/{args.model}_acc{acc:.4f}.pt')
            print(f'Saved model with accuracy: {acc:.4f}')

        cm = confusion_matrix(y_true, y_pred)
        labels_wise_acc = cm.diagonal()/cm.sum(axis=1)
        print(labels_wise_acc)
        print(confusion_matrix(y_true, y_pred))

        if use_tb:
            TB_WRITER.add_scalar('Loss/Train', running_loss/len(train_loader), epoch)
            TB_WRITER.add_scalar('Loss/Val', val_loss/len(test_loader), epoch)
            TB_WRITER.add_scalar('Acc/Train', acc_mean/len(train_loader), epoch)
            TB_WRITER.add_scalar('Acc/Val', val_acc_mean/len(test_loader), epoch)
