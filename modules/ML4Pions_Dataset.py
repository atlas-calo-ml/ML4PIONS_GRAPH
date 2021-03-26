import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import dgl
from dgl import backend as F

from torch.utils.data import Dataset, DataLoader, Sampler

from modules.fixed_radius_graph import FixedRadiusNNGraph
from dgl.geometry.pytorch import FarthestPointSampler


# --- K_NN graph dataclass ----------------#
class MLPionsDataset_KNN(Dataset):
    def __init__(self, filename, k_val, cluster_var, num_ev=-1):
        
        self.k = k_val
        self.f = uproot.open(filename)
        self.cluster_var = cluster_var
        
        self.ev_tree =  self.f['EventTree;1']
        self.cell_geo_tree = self.f['CellGeo;1']
        
        if(num_ev == -1) : 
            self.n_events = self.ev_tree.n_events
        else : 
            self.n_events = num_ev

        self.cluster_var_dict_musig = {}

        for ivar in self.cluster_var : 
            var = np.concatenate(self.ev_tree[ivar].array(library='np'))
            self.cluster_var_dict_musig[ivar] =  {'mean' : var.mean(), 'std' : var.std()}
        
        self.cell_positions, self.id_to_position,\
        self.id_to_deta, self.id_to_dphi = self.get_all_cells()
        
        self.all_graphs = []


        for i in tqdm(range( self.n_events )):
            event_dict, ev_flag = self.get_single_event(i)
            if(ev_flag == 1) : 
                self.all_graphs.append(event_dict)
    
        self.n_eff = len(self.all_graphs)
    
    def get_single_event(self,event_idx):

        # ------- building the cluster graph ---------- #
        grad_bool = False
        
        cluster_cell_ID = self.ev_tree['cluster_cell_ID'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        cluster_cell_E = self.ev_tree['cluster_cell_E'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]

        n_clusters = len(cluster_cell_ID)
        ev_flag = 1
        
        if(n_clusters == 0) : 
            #print('Empty cluster event')
            ev_flag = -1
            return {'gr' : [dgl.rand_graph(2,1)], 'truth_E' : torch.tensor([-1.])}, ev_flag

        cluster_var_dict = {}

        for ivar in self.cluster_var : 
            cluster_var_dict[ivar] =\
             ( torch.tensor(self.ev_tree[ivar].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0], requires_grad=grad_bool)
                - self.cluster_var_dict_musig[ivar]['mean']
                )/self.cluster_var_dict_musig[ivar]['std']
            
        #print('Event = ', event_idx, ', ncluster = ', n_clusters)    

        graph_list = []
        # ---- loop over clusters ---- #
        for ic in range(n_clusters) : 
            
            cell_E = np.array(cluster_cell_E[ic])
            cell_idx = np.array(cluster_cell_ID[ic])
            
            cluster_cell_pos = torch.tensor([self.id_to_position[x] for x in cell_idx], requires_grad=grad_bool)
            cluster_cell_deta = torch.tensor([self.id_to_deta[x] for x in cell_idx], requires_grad=grad_bool)
            cluster_cell_dphi = torch.tensor([self.id_to_dphi[x] for x in cell_idx], requires_grad=grad_bool)
            
            n_part = len(cluster_cell_pos)
            
            if(n_part < self.k) : 
                knn_g = dgl.knn_graph(cluster_cell_pos , n_part)
            else : 
                knn_g = dgl.knn_graph(cluster_cell_pos , self.k)
                
            knn_g.ndata['x'] = cluster_cell_pos
            #knn_g.ndata['en'] = torch.tensor(cell_E)
            cell_E = torch.tensor(cell_E, requires_grad=grad_bool)

            # for ivar in cluster_var : 
            #     print( ivar + ' shape : ', cluster_var_dict[ivar][ic] )
            clus_var = [ dgl.broadcast_nodes(knn_g, torch.reshape( cluster_var_dict[ivar][ic] , (1, 1) ) ) for ivar in self.cluster_var ]
            
            clus_var.append( torch.reshape(cluster_cell_deta, (cluster_cell_deta.shape[0], 1) ) )
            clus_var.append( torch.reshape(cluster_cell_dphi, (cluster_cell_dphi.shape[0], 1) ) )

            clus_var.insert(0, torch.reshape(cell_E, (cell_E.shape[0], 1) ) )

            knn_g.ndata['en'] = torch.stack(clus_var, dim=2)
                
            graph_list.append(knn_g)
        # -------- #
        # if( len(graph_list) == 0 ) : 
        #     print('Empty graph, Event : ', event_idx)
        cluster_energy_truth = self.ev_tree['cluster_ENG_CALIB_TOT'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        # ---------------------------------------------------------------- #
        return {'gr' : graph_list, 'truth_E' : torch.tensor(cluster_energy_truth) }, ev_flag
        
    
    def get_all_cells(self):
        rperp = self.cell_geo_tree['cell_geo_rPerp'].array(library='np')[0]
        cell_eta = self.cell_geo_tree['cell_geo_eta'].array(library='np')[0]
        cell_theta = 2*np.arctan( np.exp(-cell_eta) )
        cell_phi = self.cell_geo_tree['cell_geo_phi'].array(library='np')[0]

        cell_deta = self.cell_geo_tree['cell_geo_deta'].array(library='np')[0]
        cell_dphi = self.cell_geo_tree['cell_geo_dphi'].array(library='np')[0]

        cell_x = rperp*np.cos(cell_phi)
        cell_y = rperp*np.sin(cell_phi)
        cell_z = rperp/np.tan(cell_theta)
        cell_positions = np.column_stack([cell_x,cell_y,cell_z])
        cell_geo_ID = self.cell_geo_tree['cell_geo_ID'].array(library='np')[0]

        id_to_position = {c_id : pos  for c_id,pos in zip(cell_geo_ID,cell_positions)}
        id_to_deta     = {c_id : deta for c_id,deta in zip(cell_geo_ID,cell_deta)}
        id_to_dphi     = {c_id : dphi for c_id,dphi in zip(cell_geo_ID,cell_dphi)}

        return cell_positions, id_to_position, id_to_deta, id_to_dphi
        
    def __len__(self):

            #if(self.num_ev == -1) : 
            return self.n_eff 
            # else : 
            #     return self.num_ev


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
            
            cluster_cell_pos = torch.tensor([self.id_to_position[x] for x in cell_idx], requires_grad=True)
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
            gr_frn.ndata['en'] = torch.tensor(cell_E, requires_grad=True)
                
            graph_list.append(gr_frn)
        # -------- #
        if( len(graph_list) == 0 ) : 
            print('Empty graph, Event : ', event_idx)
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


def collate_graphs(event_list) : 
    
    n_batch = len(event_list)
    full_gr_list = []
    energy_list = []
    
    for ib in range(n_batch) : 
       
        gr_list = event_list[ib]['gr']
        tr_elist = event_list[ib]['truth_E']
        #energy_list.append(event_list[ib]['truth_E'])
        
        for i in range(len(gr_list)) : 
            #if( len(gr_list[i].ndata) !=0 ) : 
            full_gr_list.append(gr_list[i])
            energy_list.append(tr_elist[i])
                  
    return dgl.batch(full_gr_list), torch.tensor(energy_list)


