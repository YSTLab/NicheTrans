import sklearn.neighbors

import numpy as np
import pandas as pd
import scanpy as sc

from collections import defaultdict

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


class Breast_cancer(object):
    def __init__(self, adata_path, coordinate_path, ct_path):

        adata = sc.read_h5ad(adata_path)
        coordinates = pd.read_csv(coordinate_path, compression='gzip')
        ct = pd.read_excel(ct_path, sheet_name='Xenium R1 Fig1-5 (supervised)') 

        #######
        adata.obs['cell_CD20_mean'] = np.log(adata.obs['cell_CD20_mean'].values + 1)
        adata.obs['cell_HER2_mean'] = np.log(adata.obs['cell_HER2_mean'].values + 1)

        sc.pp.normalize_total(adata, target_sum=1e3)
        sc.pp.log1p(adata)
        #######

        adata.obs['x'], adata.obs['y'] = coordinates['x_centroid'].values, coordinates['y_centroid'].values
        adata.obs['ct'] = ct['Cluster'].values

        self.cell_mask = np.array(sorted(set(ct['Cluster'].values.tolist())))

        centra = adata.obs['x'].values.max()//2
        training_adata = adata[ adata.obs['x'].values <= centra ]
        testing_adata = adata[ adata.obs['x'].values >= centra ]

        ########
        self.training = self._process_data(training_adata)
        self.testing = self._process_data(testing_adata)

        self.rna_length, self.protein_length = 313, 2
        self.target_panel = np.array(['CD20', 'HER2'])
       
        num_training_spots = len(self.training)
        num_testing_spots = len(self.testing) 

        print("=> AD Mouse loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # num | ")
        print("  ------------------------------")
        print("  train    |  {:5d} spots, {} positive CD20, {} positive HER2 ".format(num_training_spots, (training_adata.obs['cell_CD20_mean'].values > 0).sum(), (training_adata.obs['cell_HER2_mean'].values > 0).sum()))
        print("  test     |  {:5d} spots, {} positive CD20, {} positive HER2 ".format(num_testing_spots, (testing_adata.obs['cell_CD20_mean'].values > 0).sum(), (testing_adata.obs['cell_HER2_mean'].values > 0).sum()))
        print("  ------------------------------")


    def _process_data(self, rna_adata):
        dataset = []
        
        ######
        rna_array = rna_adata.X.toarray()
        ct_array = rna_adata.obs['ct'].values
        indexes = rna_adata.obs.index.tolist()
        proteins = rna_adata.obs[['cell_CD20_mean', 'cell_HER2_mean']].values.astype(np.float32)
        ######
        graph = Cal_Spatial_Net_row_col(rna_adata, k_cutoff=12, model='KNN', mouse=True)

        ######
        dict_rna, dict_ct = {}, {}
        for i, index in enumerate(indexes):
            dict_rna[index] = rna_array[i]
            dict_ct[index] = (ct_array[i] == self.cell_mask) * 1
        #######
        
        for i in range(rna_adata.shape[0]):
            rna_neighbor, ct_neighbor = [], []

            index = indexes[i]
            rna, protein, ct = rna_array[i], proteins[i], dict_ct[index]

            for j in graph[index]:
                rna_neighbor.append(dict_rna[j])
                ct_neighbor.append(dict_ct[j])

            rna_neighbor = np.array(rna_neighbor)
            ct_neighbor = np.array(ct_neighbor)

            dataset.append((rna, protein, ct, rna_neighbor, ct_neighbor, index))

        return dataset

    
if __name__ == '__main__':
    dataset = Breast_cancer()
