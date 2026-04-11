import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

def read_image(img_path):
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class SMA_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors, sample = self.dataset[index]
        
        img = read_image(img_path=img_path)
        img = self.transform(img)

        rna_temp = torch.Tensor(rna_temp)
        msi_temp = torch.Tensor(msi_temp)
       
        rna_neighbors = torch.Tensor(rna_neighbors)
        msi_neighbors = torch.Tensor(msi_neighbors)

        return img, rna_temp, msi_temp, rna_neighbors, msi_neighbors, sample


class Lymph_node_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rna_temp, protein_temp, rna_neighbors, sample = self.dataset[index]
    
        rna_temp = torch.Tensor(rna_temp)
        protein_temp = torch.Tensor(protein_temp)
       
        rna_neighbors = torch.Tensor(rna_neighbors)

        return rna_temp, protein_temp, rna_neighbors, sample
    

class Breast_cancer_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rna, protein, ct, rna_neighbors, ct_neighbor, sample = self.dataset[index]
        rna_temp = torch.Tensor(rna)
        protein_temp = torch.Tensor(protein)
       
        rna_neighbors = torch.Tensor(rna_neighbors)

        return rna_temp, protein_temp, rna_neighbors, sample


class AD_Mouse_loader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        rna, protein, cell, rna_neighbor, cell_neighbor, sample = self.dataset[index]
        rna, protein, cell, rna_neighbor, cell_neighbor = torch.Tensor(rna), torch.Tensor(protein), torch.Tensor(cell), torch.Tensor(rna_neighbor), torch.Tensor(cell_neighbor)

        return rna, protein, cell, rna_neighbor, cell_neighbor, sample


class Embryonic_mouse_brain(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source_temp, target_temp, source_neighbors, sample = self.dataset[index]
    
        source_temp = torch.Tensor(source_temp)
        target_temp = torch.Tensor(target_temp)
       
        source_neighbors = torch.Tensor(source_neighbors)

        return source_temp, target_temp, source_neighbors, sample