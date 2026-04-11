import numpy as np

import random
import torch

from utils.utils import AverageMeter
from utils.evaluation import evaluator


def train(model, criterion, optimizer, trainloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    losses = AverageMeter()

    for batch_idx, (rna, protein, rna_neighbors, _) in enumerate(trainloader):

        rna, protein, rna_neighbors = rna.to(device), protein.to(device), rna_neighbors.to(device)

        ############
        if random.random() > 0.7:
            mask = torch.ones((rna_neighbors.size(0), rna_neighbors.size(1), 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).to(device)
            rna_neighbors = rna_neighbors * mask
        ############

        source, target, source_neightbors = rna, protein, rna_neighbors

        outputs = model(source, source_neightbors)

        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, source.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def test(model, testloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predict_list, target_list = [], []

    with torch.no_grad():
        for _, (rna, protein, rna_neighbors, _) in enumerate(testloader):
            rna, protein, rna_neighbors = rna.to(device), protein.to(device), rna_neighbors.to(device)
            source, target, source_neightbors = rna, protein, rna_neighbors

            outputs = model(source, source_neightbors)

            predict_list.append(outputs)
            target_list.append(target)


    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list)

    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    print('Testing Set: pearson correlation {:.4f}; spearman correlation {:.4f}; rmse {:.4f}'.format(pearson_mean, spearman_mean, rmse_mean))

    return pearson_mean
