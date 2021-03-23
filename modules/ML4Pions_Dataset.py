import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import dgl
from dgl import backend as F

from torch.utils.data import Dataset, DataLoader, Sampler

from fixed_radius_graph import FixedRadiusNNGraph
from dgl.geometry.pytorch import FarthestPointSampler

# --- K_NN graph dataclass ----------------#
class MLPionsDataset_KNN(Dataset):
    def __init__(self, filename, k_val):
        
        self.k = k_val
        self.f = uproot.open(filename)
        
        self.ev_tree =  self.f['EventTree;1']
        self.cell_geo_tree = self.f['CellGeo;1']
        
        self.n_events = self.ev_tree.num_entries
        
        self.cell_positions, self.id_to_position =\
                self.get_all_cells()
        
        self.all_graphs = []

        for i in tqdm(range(self.n_events)):
            self.all_graphs.append(self.get_single_event(i))
    
     
    
    def get_single_event(self,event_idx):

        # ------- building the cluster graph ---------- #
        
        cluster_cell_ID = self.ev_tree['cluster_cell_ID'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        cluster_cell_E = self.ev_tree['cluster_cell_E'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]

        n_clusters = len(cluster_cell_ID)
        
        if(n_clusters == 0) : 
            #print('Empty cluster event')
            return {'gr' : [dgl.rand_graph(2,1)], 'truth_E' : torch.tensor([-1.])}
            

        graph_list = []
        # ---- loop over clusters ---- #
        for ic in range(n_clusters) : 
            
            cell_E = np.array(cluster_cell_E[ic])
            cell_idx = np.array(cluster_cell_ID[ic])
            
            cluster_cell_pos = np.array([self.id_to_position[x] for x in cell_idx])
            
            n_part = len(cluster_cell_pos)
            
            if(n_part < self.k) : 
                knn_g = dgl.knn_graph(torch.tensor(cluster_cell_pos) , n_part)
            else : 
                knn_g = dgl.knn_graph(torch.tensor(cluster_cell_pos) , self.k)
                
            knn_g.ndata['x'] = torch.tensor(cluster_cell_pos)
            knn_g.ndata['en'] = torch.tensor(cell_E)
                
            graph_list.append(knn_g)
        # -------- #
        
        cluster_energy_truth = self.ev_tree['cluster_ENG_CALIB_TOT'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        # ---------------------------------------------------------------- #
        return {'gr' : graph_list, 'truth_E' : torch.tensor(cluster_energy_truth) }
        
    
    def get_all_cells(self):
        rperp = self.cell_geo_tree['cell_geo_rPerp'].array(library='np')[0]
        cell_eta = self.cell_geo_tree['cell_geo_eta'].array(library='np')[0]
        cell_theta = 2*np.arctan( np.exp(-cell_eta) )
        cell_phi = self.cell_geo_tree['cell_geo_phi'].array(library='np')[0]

        cell_x = rperp*np.cos(cell_phi)
        cell_y = rperp*np.sin(cell_phi)
        cell_z = rperp/np.tan(cell_theta)
        cell_positions = np.column_stack([cell_x,cell_y,cell_z])
        cell_geo_ID = self.cell_geo_tree['cell_geo_ID'].array(library='np')[0]

        id_to_position = {c_id : pos for c_id,pos in zip(cell_geo_ID,cell_positions)}

        return cell_positions,id_to_position
        
    def __len__(self):

             return self.n_events 


    def __getitem__(self, idx):

            return self.all_graphs[idx] 


# --- Fixed Radius graph dataclass ----------------#
class MLPionsDataset_FixedR(Dataset):
    def __init__(self, filename, radius, n_neighbor):
        
        self.R = radius
        self.n_neighbor = n_neighbor
        
        self.f = uproot.open(filename)
        
        self.ev_tree =  self.f['EventTree;1']
        self.cell_geo_tree = self.f['CellGeo;1']
        
        self.n_events = self.ev_tree.num_entries
        
        self.cell_positions, self.id_to_position =\
                self.get_all_cells()
        
        self.all_graphs = []

        for i in tqdm(range(self.n_events)):
            self.all_graphs.append(self.get_single_event(i))
    
     
    
    def get_single_event(self,event_idx):

        # ------- building the cluster graph ---------- #
        
        cluster_cell_ID = self.ev_tree['cluster_cell_ID'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        cluster_cell_E = self.ev_tree['cluster_cell_E'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]

        n_clusters = len(cluster_cell_ID)
        
        if(n_clusters == 0) : 
            #print('Empty cluster event')
            return {'gr' : [dgl.rand_graph(2,1)], 'truth_E' : torch.tensor([-1.])}
            

        graph_list = []
        # ---- loop over clusters ---- #
        for ic in range(n_clusters) : 
            
            cell_E = np.array(cluster_cell_E[ic])
            cell_idx = np.array(cluster_cell_ID[ic])
            
            cluster_cell_pos = torch.tensor([self.id_to_position[x] for x in cell_idx])
            cluster_cell_pos = torch.reshape( cluster_cell_pos, (1, cluster_cell_pos.shape[0], cluster_cell_pos.shape[1]) )
            
            
            n_part = len(cluster_cell_pos[0])
            
            if(n_part < 2) : continue
                
            if(n_part < self.n_neighbor) : 
                graph_frn = FixedRadiusNNGraph(radius = self.R, n_neighbor = n_part)
            else:
                graph_frn = FixedRadiusNNGraph(radius = self.R, n_neighbor = self.n_neighbor) 
        
            fps = FarthestPointSampler( n_part )
            centroids = fps(cluster_cell_pos)
            
            gr_frn = graph_frn(cluster_cell_pos, centroids)
            
            gr_frn.ndata['x'] = cluster_cell_pos[0]
            gr_frn.ndata['en'] = torch.tensor(cell_E)
                
            graph_list.append(gr_frn)
        # -------- #
        
        cluster_energy_truth = self.ev_tree['cluster_ENG_CALIB_TOT'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        # ---------------------------------------------------------------- #
        return {'gr' : graph_list, 'truth_E' : torch.tensor(cluster_energy_truth) }
        
    
    def get_all_cells(self):
        rperp = self.cell_geo_tree['cell_geo_rPerp'].array(library='np')[0]
        cell_eta = self.cell_geo_tree['cell_geo_eta'].array(library='np')[0]
        cell_theta = 2*np.arctan( np.exp(-cell_eta) )
        cell_phi = self.cell_geo_tree['cell_geo_phi'].array(library='np')[0]

        cell_x = rperp*np.cos(cell_phi)
        cell_y = rperp*np.sin(cell_phi)
        cell_z = rperp/np.tan(cell_theta)
        cell_positions = np.column_stack([cell_x,cell_y,cell_z])
        cell_geo_ID = self.cell_geo_tree['cell_geo_ID'].array(library='np')[0]

        id_to_position = {c_id : pos for c_id,pos in zip(cell_geo_ID,cell_positions)}

        return cell_positions,id_to_position
        
    def __len__(self):

             return self.n_events 


    def __getitem__(self, idx):

            return self.all_graphs[idx] 