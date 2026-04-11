import random
import torch

from utils.evaluation import evaluator
from utils.utils import AverageMeter


def train(model, criterion, optimizer, trainloader, use_img=True, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()     #nn.Module内置方法 将模型切换到训练模式
    losses = AverageMeter()

    for batch_idx, (imgs, source, target, source_neightbors, _, _) in enumerate(trainloader):

        source, target, source_neightbors = source.to(device), target.to(device), source_neightbors.to(device)
        ############
        if random.random() > 0.7:
            mask = torch.ones((source_neightbors.size(0), 8, 1))
            mask = torch.bernoulli(torch.full(mask.shape, 0.5)).to(device)
            source_neightbors = source_neightbors * mask
        ############

        if use_img == True:
            imgs = imgs.to(device)
            outputs = model(imgs, source, source_neightbors)
        else:
            outputs = model(source, source_neightbors)

        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data, imgs.size(0))

        if (batch_idx+1) == len(trainloader):
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))


def test(model, testloader, use_img=True, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    predict_list, target_list = [], []
    coordinates = []

    with torch.no_grad():
        for _, (imgs, source, target, source_neightbors, _, samples) in enumerate(testloader):

            source, target, source_neightbors = source.to(device), target.to(device), source_neightbors.to(device)

            if use_img == True:
                imgs = imgs.to(device)
                outputs = model(imgs, source, source_neightbors)
            else:
                outputs = model(source, source_neightbors)

            predict_list.append(outputs)
            target_list.append(target)
            coordinates += samples


    pearson_sample_list, spearman_sample_list, rmse_list = evaluator(predict_list, target_list)
    pearson_mean, spearman_mean, rmse_mean = pearson_sample_list.mean(), spearman_sample_list.mean(), rmse_list.mean()
    pearson_std, spearman_std, rmse_std = pearson_sample_list.std(), spearman_sample_list.std(), rmse_list.std()

    print('Testing Set: pearson correlation {:.4f} + {:.4f}; spearman correlation {:.4f} + {:.4f}; rmse {:.4f} + {:.4f}'
                                            .format(pearson_mean, pearson_std, spearman_mean, spearman_std, rmse_mean, rmse_std))

    return pearson_mean
