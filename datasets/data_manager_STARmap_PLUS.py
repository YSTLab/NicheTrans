import os

import numpy as np
import pandas as pd
import scanpy as sc

import sklearn.neighbors
from collections import defaultdict


# return the neighborhood nodes 
def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, mouse=False):
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    if mouse == True:
        coor = pd.DataFrame(np.stack([adata.obs['x'], adata.obs['y']], axis=1))
    else:
        coor = pd.DataFrame(np.stack([adata.obs['array_row'], adata.obs['array_col']], axis=1))

    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            # breakpoint()
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

    temp_dic = defaultdict(list)

    if mouse == True:
        for i in range(Spatial_Net.shape[0]):
            center = Spatial_Net.iloc[i, 0]
            side = Spatial_Net.iloc[i, 1]
            temp_dic[center].append(side)
    else:
        for i in range(Spatial_Net.shape[0]):
            center = Spatial_Net.iloc[i, 0]
            side = Spatial_Net.iloc[i, 1]
            center_name = str(adata.obs['array_row'][center]) + '_' + str(adata.obs['array_col'][center])
            side_name = str(adata.obs['array_row'][side]) + '_' + str(adata.obs['array_col'][side])
            temp_dic[center_name].append(side_name)

    return temp_dic


class AD_Mouse(object):
    def __init__(self, AD_adata_path, Wild_type_adata_path, n_top_genes=3000, testing_control=False):

        training_slides = ['13months-disease-replicate_1_random.h5ad']
        testing_slides = ['13months-disease-replicate_2_random.h5ad']

        self.cell_type = 'ct_top'
        adata_list = []

        for slide in training_slides + testing_slides:
            path = os.path.join(AD_adata_path, slide)
            adata_temp = sc.read_h5ad(path)

            if 'highly_variable' not in adata_temp.var.columns:
                sc.pp.highly_variable_genes(adata_temp, flavor="seurat_v3", n_top_genes=n_top_genes)
            adata_list.append(adata_temp)

        self.rna_mask = adata_list[0].var['highly_variable'].values & adata_list[1].var['highly_variable'].values 
        self.cell_mask = np.array( sorted( set(adata_list[0].obs[self.cell_type].values) & set(adata_list[1].obs[self.cell_type].values)))

        val_adata = [sc.read_h5ad( os.path.join(Wild_type_adata_path, 'spatial_13months-control-replicate_1.h5ad'))]
        val_slides = ['spatial_13months-control-replicate_1.h5ad']
        
        self.training, training_tau, training_Aβ, _ = self._process_data(adata_list[0: len(training_slides)], training_slides)
        self.testing, testing_tau, testing_Aβ, graph = self._process_data(adata_list[len(training_slides): ], testing_slides)
        self.val, _, _, _ = self._process_data(val_adata, val_slides, WT=True)

        self.graph = graph

        self.rna_length = self.rna_mask.sum()
        self.target_length = 2
        self.target_panel = np.array(['tau', 'plaque'])
        self.source_panel = adata_temp.var_names[self.rna_mask]
       
        num_training_spots = len(self.training)
        num_testing_spots = len(self.testing)

        print("=> AD Mouse loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  {:5d} spots, {} positive tao, {} positive plaque ".format(num_training_spots, training_tau, training_Aβ))
        print("  test     |  {:5d} spots, {} positive tao, {} positive plaque ".format(num_testing_spots, testing_tau, testing_Aβ))
        print("  ------------------------------")


    def _process_data(self, adata_list, slides, WT=False):

        tau, Aβ = 0, 0
        dataset = []

        for index, rna_adata in enumerate(adata_list):
            slide = slides[index]

            if WT == True:
                proteins = np.zeros((rna_adata.shape[0], 2))
            else:
                proteins = rna_adata.obs[['p-tau', 'Aβ']].values

            graph = Cal_Spatial_Net_row_col(rna_adata, k_cutoff=12, model='KNN', mouse=True)

            rna_array = rna_adata.X[:, self.rna_mask]

            indexes = rna_adata.obs.index.tolist()
            cell_array = rna_adata.obs[self.cell_type].values

            tau += proteins[:, 0].sum()
            Aβ += proteins[:, 1].sum()

            dict_rna = {}
            for i, index in enumerate(indexes):
                dict_rna[index] = rna_array[i]

            dict_cell = {}
            for i, index in enumerate(indexes):
                dict_cell[index] = (cell_array[i] == self.cell_mask) * 1

            for i in range(rna_adata.shape[0]):
                rna_neighbor, cell_neighbor = [], []

                cell = (cell_array[i] == self.cell_mask) * 1
                rna, protein = rna_array[i], proteins[i]
                index = indexes[i]

                for j in graph[index]:
                    rna_neighbor.append(dict_rna[j])
                    cell_neighbor.append(dict_cell[j])

                rna_neighbor = np.array(rna_neighbor)
                cell_neighbor = np.array(cell_neighbor)

                dataset.append((rna, protein, cell, rna_neighbor, cell_neighbor, slide + '/' + index))

        return dataset, tau, Aβ, graph

    
if __name__ == '__main__':
    dataset = AD_Mouse()
