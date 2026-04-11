import os

import numpy as np
import pandas as pd
import scanpy as sc

import sklearn.neighbors

from collections import defaultdict


# return the neighborhood nodes 
def Cal_Spatial_Net_row_col(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')

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
    for i in range(Spatial_Net.shape[0]):

        center = Spatial_Net.iloc[i, 0]
        side = Spatial_Net.iloc[i, 1]

        center_name = str(adata.obs['array_row'][center]) + '_' + str(adata.obs['array_col'][center])
        side_name = str(adata.obs['array_row'][side]) + '_' + str(adata.obs['array_col'][side])

        temp_dic[center_name].append(side_name)

    return temp_dic


class SMA(object):
    def __init__(self, path_img, rna_path, msi_path, n_top_genes=3000, n_top_targets=50):
        
        training_slides = ['V11L12-109_B1', 'V11L12-109_C1']
        testing_slides = ['V11L12-109_A1']

        self.path_img = path_img

        rna_adata_list, msi_adata_list = [], []
        rna_highly_variable_list, msi_highly_variable_list = [], []

        for slide in training_slides + testing_slides:

            adata_rna = sc.read_visium(os.path.join(rna_path, slide))
            adata_rna.var_names_make_unique()

            sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_top_genes)
            sc.pp.normalize_total(adata_rna, target_sum=1e4)
            sc.pp.log1p(adata_rna)

            rna_adata_list.append(adata_rna)
            rna_highly_variable_list.append(adata_rna.var['highly_variable'].values)

            adata_msi = sc.read_h5ad(os.path.join(msi_path, 'metabolite_' + slide + '.h5ad'))
            adata_msi.var_names_make_unique()
            sc.pp.highly_variable_genes(adata_msi, flavor="seurat_v3", n_top_genes=n_top_targets)
            sc.pp.log1p(adata_msi)

            msi_adata_list.append(adata_msi)
            msi_highly_variable_list.append(adata_msi.var['highly_variable'].values)


        ##############
        temp = np.concatenate([ rna_adata_list[0].X.toarray(), rna_adata_list[1].X.toarray(), rna_adata_list[2].X.toarray()], axis=0)
        self.rna_mean, self.rna_std = temp.mean(axis=0)[None, ], temp.std(axis=0)[None, ]
        
        temp_mask = (self.rna_std == 0)
        self.rna_std[temp_mask] = 1

        ###############

        self.rna_mask =  rna_highly_variable_list[0] & rna_highly_variable_list[1] & rna_highly_variable_list[2]
        self.msi_mask = msi_highly_variable_list[0] & msi_highly_variable_list[1] & msi_highly_variable_list[2]

        self.training = self._process_data(rna_adata_list[0:2], msi_adata_list[0:2], training_slides)
        self.testing = self._process_data(rna_adata_list[2:], msi_adata_list[2:], testing_slides)

        self.rna_length = (self.rna_mask * 1).sum()
        self.msi_length = (self.msi_mask * 1).sum()
        self.target_panel = adata_msi.var['metabolism'].values[self.msi_mask].tolist()  #[i for i in range(self.msi_length)]
        self.source_panel = adata_rna.var_names[self.rna_mask]

        num_training_spots, num_testing_spots = len(self.training), len(self.testing)
        num_training_slides, num_testing_slides = len(training_slides), len(testing_slides)

        ori_num_training_spots = rna_adata_list[0].shape[0] + rna_adata_list[1].shape[0]
        ori_num_testing_spots = rna_adata_list[2].shape[0]

        print("=> SMA loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  Without filtering {:5d} spots from {:5d} slides ".format(ori_num_training_spots, num_training_slides))
        print("  test     |  Without filtering {:5d} spots from {:5d} slides".format(ori_num_testing_spots, num_testing_slides))
        print("  train    |  After filting {:5d} spots from {:5d} slides ".format(num_training_spots, num_training_slides))
        print("  test     |  After filting {:5d} spots from {:5d} slides".format(num_testing_spots, num_testing_slides))
        print("  ------------------------------")


    def _dictionary_data(self, adata, rna=False):
        dictionary = {}
        array_row, array_col = adata.obs['array_row'].values, adata.obs['array_col'].values

        array = adata.X.toarray()
    
        for i in range(adata.shape[0]):
            dictionary[str(int(array_row[i])) + '_' +  str(int(array_col[i])) ] = array[i]

        return dictionary

        
    def _process_data(self, rna_adata_list, msi_adata_list, names):

        dataset = []

        for i in range(len(rna_adata_list)):

            rna_temp_adata = rna_adata_list[i]
            msi_temp_adata = msi_adata_list[i]

            rna_dic = self._dictionary_data(rna_temp_adata, rna=True)
            msi_dic = self._dictionary_data(msi_temp_adata)

            # graph = Cal_Spatial_Net_spatial_cite_seq(rna_temp_adata,  k_cutoff=8, model='KNN')
            graph_1 = Cal_Spatial_Net_row_col(rna_temp_adata,  rad_cutoff=2**(1/2), model='Radius')
            graph_2 = Cal_Spatial_Net_row_col(rna_temp_adata,  rad_cutoff=2, model='Radius')

            rna_keys, msi_keys = rna_dic.keys(), msi_dic.keys()

            for key in rna_keys:
                if key not in msi_keys:
                    continue
                else:
                    rna_temp = rna_dic[key][self.rna_mask]
                    msi_temp = msi_dic[key][self.msi_mask]

                    # quality check for rna and msi
                    if rna_temp.sum() == 0 or msi_temp.sum() == 0:
                        continue
                    else:
                        img_path = os.path.join(self.path_img, names[i], key + '.png')

                        rna_neighbors, msi_neighbors = [], []

                        neighbors_1, neighbors_2 = graph_1[key], graph_2[key]
                        neighbors_2 = [item for item in neighbors_2 if item not in neighbors_1]

                        # connect to the first round 
                        for j in neighbors_1:
                            if j not in rna_keys:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])

                            if j not in msi_keys:
                                msi_neighbors.append(np.zeros_like(msi_temp))
                            else:
                                msi_neighbors.append(msi_dic[j][self.msi_mask])

                        if len(neighbors_1) != 4:
                            for _ in range(4-len(neighbors_1)):
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                msi_neighbors.append(np.zeros_like(msi_temp))

                        # connect to the second round
                        for j in neighbors_2:
                            if j not in rna_keys:
                                rna_neighbors.append(np.zeros_like(rna_temp))
                            else:
                                rna_neighbors.append(rna_dic[j][self.rna_mask])

                            if j not in msi_keys:
                                msi_neighbors.append(np.zeros_like(msi_temp))
                            else:
                                msi_neighbors.append(msi_dic[j][self.msi_mask])

                        if len(neighbors_2) != 4:
                            for _ in range(4-len(neighbors_2)):
                                rna_neighbors.append(np.zeros_like(rna_temp))
                                msi_neighbors.append(np.zeros_like(msi_temp))

                        rna_neighbors = np.stack(rna_neighbors)
                        msi_neighbors = np.stack(msi_neighbors)

                        dataset.append((img_path, rna_temp, msi_temp, rna_neighbors, msi_neighbors, names[i] + '/' + key))

        return dataset

if __name__ == '__main__':
    dataset = SMA()
