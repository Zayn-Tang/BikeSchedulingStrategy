# -*- coding:utf-8 -*-

from model.STSAGCN import STSAGCN
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn

import json
from utils.params import get_params
from utils.dataloader import CikybikeDataset
from utils.tools import save_model, load_model, get_adjacency_matrix, construct_adj
from utils.metrics import masked_mae_test, masked_rmse_test

SAVE_PATH = Path('./SaveModel/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

args = get_params()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(args)


def train_one_epoch(epoch, model, criterion, data_loader, optimizer, device="cuda:0"):
    loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)
    RMSE_score = torch.tensor(0.).to(device)
    MAE_score = torch.tensor(0.).to(device)
    model.train()

    for step, (x, y) in enumerate(data_loader):
        # x: [8, 6, 330, 1]   y: [8, 3, 330, 1]
        optimizer.zero_grad()
        x = x.to(device)
        y = y.squeeze(-1).to(device)
        pred = model(x)

        loss = criterion(pred, y)
        loss_all += loss.item()
        loss.backward()
        RMSE_score += masked_rmse_test(y.cpu(), pred.cpu())
        MAE_score += masked_mae_test(y.cpu(), pred.cpu())
        count += 1

        if step % 200 == 0:
            print("Train Step: ", step, " | Loss:", (loss_all/count).item(), " | RMSE: ", (RMSE_score/count).item(), "| MAE: ", (MAE_score/count).item())
        
        optimizer.step()

    return (RMSE_score/count).item(), (MAE_score/count).item()


def evaluate(model, criterion, data_loader, device="cuda:0"):
    loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)
    RMSE_score = torch.tensor(0.).to(device)
    MAE_score = torch.tensor(0.).to(device)

    model.eval()
    for step, (x, y) in enumerate(data_loader):
        # x: [8, 6, 330, 1]   y: [8, 3, 330, 1]
        x = x.to(device)
        y = y.squeeze(-1).to(device)
        with torch.no_grad():
            pred = model(x)

        loss = criterion(pred, y)
        RMSE_score += masked_rmse_test(y.cpu(), pred.cpu())
        MAE_score += masked_mae_test(y.cpu(), pred.cpu())
        loss_all += loss.item()
        count += 1

        if step % 200 == 0:
            print("Valid Step: ", step, " | Loss:", (loss_all/count).item(), " | RMSE: ", (RMSE_score/count).item(), "| MAE: ", (MAE_score/count).item())

    return (RMSE_score/count).item(), (MAE_score/count).item()


def train(args):
    model = STSAGCN(adj_mx, args.num_of_history, args.num_of_vertices, 1, [[64, 64, 64], [64, 64, 64]], 64, 64, "relu", True, True, True, args.num_of_predict, 3).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    # criterion = nn.L1Loss(reduction="mean")
    criterion = nn.MSELoss(reduction="mean")

    start_epoch, model, _ = load_model(model, optimizer, path=SAVE_PATH / 'latest.pt')
    min_rmse = np.inf
    # min_mae = np.inf
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_RMSE_score, train_MAE_score = train_one_epoch(epoch, model, criterion, train_loader, optimizer)

        valid_RMSE_score, valid_MAE_score = evaluate(model, criterion, valid_loader)

        print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']} | Train RMSE: {train_RMSE_score} | Train MAE: {train_MAE_score}" ,
              f"| Valid RMSE: {valid_RMSE_score} | Valid MAE: {valid_MAE_score} "
            )

        save_model(model, epoch + 1, optimizer=optimizer, path= SAVE_PATH / 'latest.pt')
        if valid_RMSE_score < min_rmse:
            min_rmse = valid_RMSE_score
            save_model(model, path=SAVE_PATH / f'RMSE_{min_rmse:.3f}_best.pt', only_model=True)

        scheduler.step(valid_RMSE_score)

if __name__ == '__main__':
    torch.manual_seed(2023)
    np.random.seed(2023)
    cudnn.benchmark = True

    train_dataset = np.load(r".\data\DataBase\train.npz")
    valid_dataset = np.load(r".\data\DataBase\valid.npz")
    train_dataset = CikybikeDataset(train_dataset, args)
    valid_dataset = CikybikeDataset(valid_dataset, args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True ,drop_last=True, num_workers=0, pin_memory = True )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False ,drop_last=True, num_workers=0, pin_memory = True )

    adj_mx, distance_mx = get_adjacency_matrix(args.adj_filename, args.num_of_vertices, args.id_filename)
    adj_mx = construct_adj(adj_mx, args.adj_steps)
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(args.device)


    train(args)



