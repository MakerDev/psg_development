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

from models.stagenet import StageNet_DCNN_SKIPLSTM
from torch.nn.utils.rnn import pad_sequence
from utils.transforms import *

class OnTheFlyArousalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_list, num_channels, transforms = None, eval=False):
        super().__init__()

        self.data_dir = data_dir
        self.file_list = file_list
        self.num_channels = num_channels
        self.transforms = transforms
        self.eval = eval
        self.cache = {}

    def __len__(self):
        return len(self.file_list)        
        
    def __getitem__(self, idx):
        if self.eval and idx in self.cache:
            x, y = self.cache[idx]
        else:
            x, y = self.load_labeled_data(self.file_list[idx])

        if self.eval and idx not in self.cache:
            self.cache[idx] = (x, y)

        
        return x, y

    def load_labeled_data(self, filename):
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            d = pickle.load(f)
            x, y = d['x'], d['y'].astype(np.int64)
            if self.num_channels != 9:
                x = x[:, :, :self.num_channels]            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
        return x, y

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def collate_fn(batch):
    data, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in data])  # Time_steps per sequence
    padded_data = pad_sequence(data, batch_first=True)  # Pad sequences to (Batch, Max_Time_steps, Length, Channels)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # Pad labels with -1 as ignored index
    return padded_data, padded_labels, lengths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_tb', type=str2bool, default=False, help='Use tensorboard')
    parser.add_argument("--lr", type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--loss_weight', type=str, default='dynamic', help='Loss weight')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    use_tb = args.use_tb
    tb_name = f'{str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_STAGENET_LR{args.lr}_{args.tag}'
    if use_tb:
        TB_WRITER = tb.SummaryWriter(f'/home/honeynaps/data/tensorboards/{tb_name}')
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_dir = '/home/honeynaps/data/dataset/PICKLE/SLEEP_50'
    train_data_list  = []
    train_label_list = []
    val_data_list    = []
    val_label_list   = []

    file_names = natsort.natsorted(os.listdir(dataset_dir))
    random_indices = np.random.permutation(len(file_names))
    file_names = [file_names[i] for i in random_indices]

    train_files = file_names[:int(0.8*len(file_names))]
    val_files = file_names[int(0.8*len(file_names)):]
    ext = '.pkl'

    transforms = Compose([ToTensor(), NormaliseAndAddRandNoise()])
    # Dataset and DataLoader
    train_dataset = OnTheFlyArousalDataset(dataset_dir, train_files, num_channels=9, transforms=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    
    test_dataset = OnTheFlyArousalDataset(dataset_dir, val_files, num_channels=9, transforms=transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = StageNet_DCNN_SKIPLSTM(num_signals=9)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.loss_weight == 'default':
        class_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        class_weights = [0.8149232316038625, 0.8440648499193566, 0.8740966883810561, 0.5221141157495653, 0.9448011143461594]


    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor(class_weights).to(device))
    optimizer = torch.optim.Adam(
        params       = model.parameters(), 
        lr           = args.lr,
        weight_decay = 0.0001,
        amsgrad      = False)
    
    num_epochs = 200
    for epoch in range(num_epochs):  
        model.train()
        running_loss = 0.0
        acc_mean = 0.0
        for padded_data, padded_labels, lengths in train_loader:
            padded_data = padded_data.to(device) 
            padded_labels = padded_labels.to(device) 
            
            outputs = model(padded_data) 
            outputs = outputs.reshape(-1, outputs.size(-1))  
            padded_labels = padded_labels.reshape(-1)  
            
            loss = criterion(outputs, padded_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc_mean += (outputs.argmax(dim=-1) == padded_labels).float().mean().item()
            running_loss += loss.item()
        
        y_true, y_pred = [], []
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_acc_mean = 0.0
            for padded_data, padded_labels, lengths in test_loader:
                padded_data = padded_data.to(device)
                padded_labels = padded_labels.to(device)
                
                outputs = model(padded_data)
                outputs = outputs.reshape(-1, outputs.size(-1))
                padded_labels = padded_labels.reshape(-1)
                
                loss = criterion(outputs, padded_labels)
                val_acc_mean += (outputs.argmax(dim=-1) == padded_labels).float().mean().item()
                val_loss += loss.item()

                y_true.extend(padded_labels.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=-1).cpu().numpy())

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {acc_mean/len(train_loader):.4f}, Val Acc: {val_acc_mean/len(test_loader):.4f}', end='')
        print(f"Weights: {class_weights}")
        if epoch % 5 == 0:
            cm = confusion_matrix(y_true, y_pred)
            labels_wise_acc = cm.diagonal()/cm.sum(axis=1)
            print(labels_wise_acc)
            print(confusion_matrix(y_true, y_pred))

        if use_tb:
            TB_WRITER.add_scalar('Loss/Train', running_loss/len(train_loader), epoch)
            TB_WRITER.add_scalar('Loss/Val', val_loss/len(test_loader), epoch)
            TB_WRITER.add_scalar('Acc/Train', acc_mean/len(train_loader), epoch)
            TB_WRITER.add_scalar('Acc/Val', val_acc_mean/len(test_loader), epoch)
        