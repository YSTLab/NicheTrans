import torch
import scipy
import math
import numpy as np
import warnings

import matplotlib.pyplot as plt


def draw_dot_plots(predict_list, target_list, pearson_sample_list, norm_rmse_list, panel, training=True):
    for i in range(predict_list.shape[1]):

        predict = predict_list[:, i]
        targets = target_list[:, i]

        max_val = max(max(predict), max(targets))
        lim = (0, max_val)
        plt.plot(lim, lim, color='red', linestyle='--')

        plt.figure(figsize=(8, 6))
        plt.scatter(predict, targets, color='blue', alpha=0.5)
        plt.plot(lim, lim, color='red', linestyle='--', linewidth=2)

        plt.xlim(lim)
        plt.ylim(lim)

        plt.xlabel('Predicted Number')
        plt.ylabel('Ground Truth')
        plt.title('Predicted vs. Ground Truth: Pearson {} and norm rmse {} of Panel {}'.format(str(pearson_sample_list[i])[0:5], str(norm_rmse_list[i])[0:5], panel[i]))
        plt.grid(True)
        if training==True:
            plt.savefig('./plots/training_{}_{}.jpg'.format(panel[i], str(i)))
        else:
            plt.savefig('./plots/testing_{}_{}.jpg'.format(panel[i], str(i)))
        plt.close()


def evaluator(predict_list, target_list):

    if isinstance(predict_list, list):
        predict_list = torch.cat(predict_list, dim=0).cpu().detach().numpy()
    if isinstance(target_list, list):
        target_list = torch.cat(target_list, dim=0).cpu().detach().numpy()

    pearson_sample_list, spearman_sample_list, rmse_list = [], [], []
    num_targets = target_list.shape[1]

    for i in range(num_targets):
        pearson_corr, _ = scipy.stats.pearsonr(predict_list[:, i], target_list[:, i])
        spearman_corr, _ = scipy.stats.spearmanr(predict_list[:, i], target_list[:, i])
        rmse =  np.sqrt(np.mean( (predict_list[:, i] - target_list[:, i])**2 ))

        if np.isnan(pearson_corr): continue
        else: pearson_sample_list.append(pearson_corr)

        if np.isnan(spearman_corr): continue
        else: spearman_sample_list.append(spearman_corr)
        
        rmse_list.append(rmse)

    pearson_sample_list = np.array(pearson_sample_list)
    spearman_sample_list = np.array(spearman_sample_list) 
    rmse_list = np.array(rmse_list)

    return pearson_sample_list, spearman_sample_list, rmse_list