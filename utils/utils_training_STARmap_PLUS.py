import numpy as np

import random
import torch

from utils.evaluation import evaluator

from utils.utils import AverageMeter
from sklearn.metrics import roc_auc_score, confusion_matrix


def train(model, criterion, optimizer, trainloader, ct_information=False, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    losses = AverageMeter()
    for batch_idx, (rna, protein, cell, rna_neighbors, cell_neighbor, _) in enumerate(trainloader):

        rna, protein, rna_neighbors = rna.to(device), protein.to(device), rna_neighbors.to(device)
        cell, cell_neighbor = cell.to(device), cell_neighbor.to(device)

        ############
        if random.random() > 0.7:
            mask = torch.ones((rna_neighbors.size(0), rna_neighbors.size(1), 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).to(device)
            rna_neighbors = rna_neighbors * mask
            cell_neighbor = cell_neighbor * mask
        ############

        cell_inf = torch.cat([cell[:, None, :], cell_neighbor], dim=1)
        source, target, source_neightbors = rna, protein, rna_neighbors

        if ct_information == True:
            outputs = model(source, source_neightbors, cell_inf)
        else:
            outputs = model(source, source_neightbors)
        outputs = torch.sigmoid(outputs)

        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, source.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def test(model, testloader, ct_information=False, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predict_list, target_list = [], []

    with torch.no_grad():
        for _, (source, target, cell, source_neightbors, cell_neighbor, _) in enumerate(testloader):

            source, target, source_neightbors = source.to(device), target.to(device), source_neightbors.to(device)
            cell_inf = torch.cat([cell[:, None, :], cell_neighbor], dim=1).to(device)

            if ct_information == True:
                outputs = model(source, source_neightbors, cell_inf)
            else:
                outputs = model(source, source_neightbors)
            outputs = torch.sigmoid(outputs)

            predict_list.append(outputs)
            target_list.append(target)

    predict_list = torch.cat(predict_list, dim=0).cpu().numpy()
    target_list = torch.cat(target_list, dim=0).cpu().numpy()

    ##############
    auc_tau = roc_auc_score(target_list[:, 0], predict_list[:, 0])
    tn, fp, fn, tp = confusion_matrix(target_list[:, 0], (predict_list[:, 0] > 0.5).astype(int)).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"tau AUC: {auc_tau}, tau sensitivity {sensitivity}, tay specificity {specificity}")

    ##############
    auc_plaque = roc_auc_score(target_list[:, 1], predict_list[:, 1])
    tn, fp, fn, tp = confusion_matrix(target_list[:, 1], (predict_list[:, 1] > 0.5).astype(int)).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Aβ AUC: {auc_plaque}, Aβ sensitivity {sensitivity}, Aβ specificity {specificity}")

    return None
