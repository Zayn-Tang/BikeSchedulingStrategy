# -*- coding:utf-8 -*-
from model.STSAGCN import STSAGCN
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn

from utils.params import get_params
from utils.dataloader import CikybikeDataset
from utils.tools import load_model, get_adjacency_matrix, construct_adj
from utils.metrics import masked_mae_test, masked_rmse_test

SAVE_MODEL_PATH = Path('./SaveModel/')
OUTPUT_PATH = Path('./Output/')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

args = get_params()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(args)


def inference(model, data_loader,  device="cuda:0"):
    count = torch.tensor(0.).to(device)
    RMSE_score = torch.tensor(0.).to(device)
    MAE_score = torch.tensor(0.).to(device)
    output = []

    model.eval()
    for step, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.squeeze(-1).to(device)
        # x: [8, 6, 330, 1]   y: [8, 3, 330, 1]
        with torch.no_grad():
            pred = model(x)
            output.append(pred.detach().cpu())

        RMSE_score += masked_rmse_test(y.cpu(), pred.cpu())
        MAE_score += masked_mae_test(y.cpu(), pred.cpu())
        count += 1

        if step % 500 == 0:
            print("Test Step: ", step, " | RMSE: ", (RMSE_score/count).item(), " | MAE: ", (RMSE_score/count).item())

    output = torch.concat(output, dim=0)
    np.save(OUTPUT_PATH / "output", output)
    return (RMSE_score/count).item(), (MAE_score/count).item()

if __name__ == '__main__':
    torch.manual_seed(2023)
    np.random.seed(2023)
    cudnn.benchmark = True

    adj_mx, distance_mx = get_adjacency_matrix(args.adj_filename, args.num_of_vertices, args.id_filename)
    adj_mx = construct_adj(adj_mx, args.adj_steps)
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(args.device)
    model = STSAGCN(adj_mx, args.num_of_history, args.num_of_vertices, 1, [[64, 64, 64], [64, 64, 64]], 64, 64, "relu", True, True, True, args.num_of_predict, 3).cuda()
    model = load_model(model, None, path=SAVE_MODEL_PATH / 'latest.pt', only_model=True)

    test_dataset = np.load(r".\data\DataBase\test.npz")
    test_dataset = CikybikeDataset(test_dataset, args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True ,drop_last=True, num_workers=0, pin_memory = True )

    RMSE_score, MAE_score = inference(model, test_loader)
    print(f"Test RMSE: {RMSE_score}, MAE: {MAE_score}")

