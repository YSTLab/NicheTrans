"""Attribution analysis for Tutorial 3.3 (batch mode to avoid OOM)."""
import os, sys, warnings, torch
import torch.nn as nn
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, '/mnt/datadisk0/NicheTrans')
os.chdir('/mnt/datadisk0/NicheTrans')

from captum.attr import IntegratedGradients
from model.nicheTrans_attribution_SMA import NicheTrans
from datasets.data_manager_SMA import SMA
from utils.utils_dataloader import sma_dataloader

# ── Load args ──────────────────────────────────────────────────────────────
exec(open('args/args_SMA.py').read())

# ── Dataset & dataloader ───────────────────────────────────────────────────
print("Loading dataset...")
dataset = SMA(path_img=args.path_img, rna_path=args.rna_path,
              msi_path=args.msi_path, n_top_genes=args.n_source,
              n_top_targets=args.n_target)
_, testloader = sma_dataloader(args, dataset)

# ── Model ──────────────────────────────────────────────────────────────────
device = torch.device('cpu')
source_dimension, target_dimension = dataset.rna_length, dataset.msi_length
print(f"source_dim={source_dimension}, target_dim={target_dimension}")

model = NicheTrans(source_length=source_dimension, target_length=target_dimension,
                   noise_rate=args.noise_rate, dropout_rate=args.dropout_rate)
model = model.to(device)
state_dict = torch.load('NicheTrans_SMA_last.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")

# ── Collect inputs ─────────────────────────────────────────────────────────
print("Collecting test inputs...")
input_list = []
with torch.no_grad():
    for imgs, source, target, source_neightbors, _, samples in testloader:
        source_data = torch.cat([source[:, None, :], source_neightbors], dim=1)
        input_list.append(source_data)

rna_list = torch.cat(input_list, dim=0)
print(f"rna_list shape: {rna_list.shape}")  # [N, 9, 1159]

# ── Integrated Gradients in batches (avoids OOM) ──────────────────────────
ig = IntegratedGradients(model)
BATCH = 64   # reduce if still OOM; increase for speed
N_STEPS = 20  # lower than default 50 to save memory; increase for accuracy

all_attrs = []
n = rna_list.shape[0]
for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    batch = rna_list[start:end].to(device)
    baseline = torch.zeros_like(batch)
    attrs = ig.attribute(batch, baseline, target=0, n_steps=N_STEPS)
    all_attrs.append(attrs.detach().cpu())
    print(f"  [{end}/{n}] done")

attributions = torch.cat(all_attrs, dim=0).numpy()
print(f"Attributions shape: {attributions.shape}")

# ── Top-gene ranking ───────────────────────────────────────────────────────
num_genes = 30
grad_norm = np.abs(attributions).sum(axis=0).sum(axis=0)
top_indices = np.argsort(grad_norm)[-num_genes:]
highly_correlated_genes_0 = dataset.source_panel[top_indices]
print(f"Top {num_genes} genes:\n{highly_correlated_genes_0}")

# ── Save results ───────────────────────────────────────────────────────────
np.save('attributions.npy', attributions)
np.save('grad_norm.npy', grad_norm)
print("Saved attributions.npy and grad_norm.npy")

# ── Plot ───────────────────────────────────────────────────────────────────
sorted_data = np.sort(grad_norm.squeeze())[::-1]
indices = np.arange(len(sorted_data))

fig, ax = plt.subplots(1, figsize=(6.5, 2), dpi=300)
plt.scatter(indices[0:30], sorted_data[0:30], s=2, color='#6F1D57')
plt.scatter(29 + (indices[30:] - 29) / 40, sorted_data[30:], s=0.2, color='#FCBD8B')

for i in range(num_genes):
    plt.annotate(highly_correlated_genes_0[num_genes - i - 1],
                 xy=(i, sorted_data[i]),
                 xytext=(0, 5),
                 textcoords='offset points',
                 ha='center',
                 rotation=90,
                 fontsize=4.5)

plt.fill_between(indices[0:30], sorted_data[0:30], color='#E8789A', alpha=0.4)
plt.fill_between(29 + (indices[30:] - 29) / 40, sorted_data[30:], color='#EDDAB9', alpha=0.4)

custom_xticks = [0, 30, 60]
custom_xticklabels = ['0', '30', '1230']
ax.set_xticks(custom_xticks)
ax.set_xticklabels(custom_xticklabels, fontsize=5)
plt.yticks(fontsize=5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('attribution_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved to attribution_plot.png")
