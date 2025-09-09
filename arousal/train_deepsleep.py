import os
import numpy as np
import torch
import pickle
import random
import datetime
import torch.utils.tensorboard as tb

from torch.utils.data import DataLoader
from utils.datasets import OnTheFlyArousalDataset
from utils.losses import *
from utils.score2018 import *
from models.DeepSleepNet2 import *
from models.DeepSleepNet1 import *
from models.DeepSleepSota import DeepSleepNetSota
from utils.transforms import *


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def eval_fn(model, loader, device, comp_score = True):
    model.eval()
    
    scores = Challenge2018ScoreVer2() if comp_score else None
    
    if comp_score:
        criterion = CustomBCELoss().to(device)
    else:
        criterion = CustomBCEWithLogitsLoss().to(device)
    
    with torch.no_grad():
        loss_epoch_sum = 0
        val_auroc, val_auprc, best_f1, best_th = 0, 0, 0, 0
        for x, y, idx in loader:          
            x = x.to(device = device)
            y = y.to(device = device)

            # with torch.amp.autocast(device, enabled = torch.cuda.is_available() and not comp_score):
            y_pred = model(x, comp_score)
            # y_pred = torch.sigmoid(y_pred)
            loss = criterion(y_pred, y)
        
            # compute AUROC/AUPRC score for each record in batch
            if comp_score:
                for i, single_idx in enumerate(idx):
                    record_name = str(single_idx.item())
                    y_target = y[i].view(-1).to('cpu')
                    y_pred_i = y_pred[i].view(-1).to('cpu')
                    y_pad_mask = y_target != -1
                    scores.score_record(y_target[y_pad_mask], y_pred_i[y_pad_mask], record_name)
                    auroc = scores.record_auroc(record_name)
                    auprc = scores.record_auprc(record_name)
                    f1 = scores.record_f1(record_name)
                    th = scores.record_best_threshold(record_name)
                    # print(f"Record{record_name}  AUROC: {auroc},  AUPRC: {auprc}")
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


def load_labeled_data(file_paths, num_channels = 9):
    data_list, labels = [], []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
            x, y = d['x'][:num_channels,:], d['y'].astype(np.int64)
            data_list.append(x)
            labels.append(y)

    return data_list, labels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DeepSleepSota')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_channels', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='asl')
    parser.add_argument('--freq', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--use_tb', type=str2bool, default=False)
    parser.add_argument('--mix_db', type=str2bool, default=False)
    parser.add_argument('--target_db', type=str, default='2', choices=['1', '2', 'mixed'])
    parser.add_argument('--ver', type=int, default=3)
    parser.add_argument('--noise_minmax', type=str, default='0.5_0.8')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

    if args.mix_db:
        dataset_dirs = [f"/home/honeynaps/data/dataset/PICKLE/AROUSAL_VER{args.ver}_{args.freq}_PAD",
                        f"/home/honeynaps/data/dataset2/PICKLE/AROUSAL_VER{args.ver}_{args.freq}_PAD"]
    else:
        db_option = ''

        if args.target_db == '2':
            db_option = '2'
        elif args.target_db == 'mixed':
            db_option = '_mixed'

        dataset_dirs = [f"/home/honeynaps/data/dataset{db_option}/PICKLE/AROUSAL_VER{args.ver}_{args.freq}_PAD"]

    # Load labeled data files
    labels_file_paths = []
    for dataset_dir in dataset_dirs:
        labels_file_paths.extend([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.pkl')])

    random.shuffle(labels_file_paths)
    train_ratio = 0.8
    split_index = int(train_ratio*len(labels_file_paths))
    train_files = labels_file_paths[:split_index]
    val_files = labels_file_paths[split_index:]

    transforms = ["NormaliseAndAddRandNoise"]
    model_in_channels = 3 if 'SelectOne' in transforms else args.num_channels
    val_transforms = ["NormaliseOnly", "SelectOneEval"] if "SelectOne" in transforms else ["NormaliseOnly"]
    tf_str = "_".join(transforms)

    transforms = build_transforms(transforms, n_channels = args.num_channels, noise_minmax = args.noise_minmax)
    val_transforms = build_transforms(val_transforms, n_channels = args.num_channels, noise_minmax = args.noise_minmax)
    
    train_dataset = OnTheFlyArousalDataset(train_files, args.num_channels, transforms=transforms)
    val_dataset = OnTheFlyArousalDataset(val_files, args.num_channels, val_transforms)

    mix_str = 'True' if args.mix_db else args.target_db
    current_setting = f"{args.model}_FS{args.freq}_{args.loss}_CH{args.num_channels}_BS{args.batch_size}_{tf_str}_LR{args.lr}" + \
                      f"_MIX{args.mix_db}_VER{args.ver}_NS_{args.noise_minmax}_{args.tag}"

    tb_name = f'{str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{current_setting}'
    if args.use_tb:
        TB_WRITER = tb.SummaryWriter(f'/home/honeynaps/data/eis/arousalnet_r1/tensorboards/{tb_name}')

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    if args.model.lower() == 'deepsleep2':
        model = DeepSleepNet2(in_channels=model_in_channels, linear=True).to(device)
    else:
        model = DeepSleepNetSota(n_channels=model_in_channels).to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.loss == 'bce':
        criterion = CustomBCEWithLogitsLoss().to(device)
    elif args.loss == 'asl':
        criterion = CustomAsymmetricLoss().to(device)
    elif args.loss == 'ba_asl':
        criterion = BoundaryAwareAsymmetricLoss().to(device)
    else:
        criterion = dice_coef_loss

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-5)

    num_epochs = 200
    best_train_auprc, best_val_auprc = 0, 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}: {current_setting}")
        model.train()

        for i, (x, y, idx) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
        train_loss, train_auroc, train_auprc, train_th = eval_fn(model, train_dataloader, device, comp_score = True)
        print(f"Train loss: {train_loss:.4f}, AUROC: {train_auroc:.4f}, AUPRC: {train_auprc:.4f},")

        if train_auprc > best_train_auprc:
            best_train_auprc = train_auprc

        val_loss, val_auroc, val_auprc, val_th = eval_fn(model, val_dataloader, device, comp_score = True)
        print(f"Validation loss: {val_loss:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")

        print(f"Best Train AUPRC: {best_train_auprc:.4f} Best Val AUPRC: {best_val_auprc:.4f}")
        print("=====================================================")

        if args.use_tb:
            TB_WRITER.add_scalar('Loss/train', train_loss, epoch)
            TB_WRITER.add_scalar('Loss/val', val_loss, epoch)
            TB_WRITER.add_scalar('AUROC/train', train_auroc, epoch)
            TB_WRITER.add_scalar('AUROC/val', val_auroc, epoch)
            TB_WRITER.add_scalar('AUPRC/train', train_auprc, epoch)
            TB_WRITER.add_scalar('AUPRC/val', val_auprc, epoch)

