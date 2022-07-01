import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import dgl
from dgl import backend as F

from torch.utils.data import Dataset, DataLoader, Sampler

#from modules.fixed_radius_graph import FixedRadiusNNGraph
#from dgl.geometry.pytorch import FarthestPointSampler

layer_name = {'PreSamplerB' : [0, 1540.00],
              'PreSamplerE' : [1, 1540.00],
              'EMB1' : [0, 1532.18],
              'EMB2' : [0, 1723.89],
              'EMB3' : [0, 1923.02], 
              'EME1' : [1, 3790.03],
              'EME2' : [1, 3983.68],
              'EME3' : [1, 4195.84],
              'HEC0' : [1, 4461.25],
              'HEC1' : [1, 4869.50],
              'HEC2' : [1, 5424.50],
              'HEC3' : [1, 5905.00],
              'TileBar0' : [0, 2450.00],
              'TileBar1' : [0, 2995.00],
              'TileBar2' : [0, 3630.00],
              'TileGap1' : [0, 3215.00],
              'TileGap2' : [0, 3630.00],
              'TileGap3' : [0, 2246.50],
              'TileExt0' : [0, 2450.00],
              'TileExt1' : [0, 2870.00],
              'TileExt2' : [0, 3480.00]
             }


cluster_var_dict_musig = {
    
    'cluster_EM_PROBABILITY'  :  {'mean': 0.13739075, 'std': 0.23810464} ,
    'cluster_HAD_WEIGHT'  :  {'mean': 1.0977454, 'std': 0.105446674} ,
    'cluster_OOC_WEIGHT'  :  {'mean': 1.3992951, 'std': 0.43550345} ,
    'cluster_DM_WEIGHT'  :  {'mean': 1.1528628, 'std': 0.2928971} ,
    'cluster_CENTER_MAG'  :  {'mean': 3703.2527, 'std': 1174.078} ,
    'cluster_FIRST_ENG_DENS'  :  {'mean': 1.7587054e-05, 'std': 8.9226814e-05} ,
    'cluster_CENTER_LAMBDA'  :  {'mean': 864.2519, 'std': 821.3119} ,
    'cluster_ISOLATION'  :  {'mean': 0.8162972, 'std': 0.2107719} ,
    'trackPt'  :  {'mean': 201.44667, 'std': 1793.1217} ,
    'trackP'  :  {'mean': 333.57788, 'std': 2367.859} ,
    'trackMass'  :  {'mean': 0.12962861, 'std': 0.4628284} ,
    'trackEta'  :  {'mean': 0.0128200175, 'std': 1.365982} ,
    'trackPhi'  :  {'mean': -0.0047495714, 'std': 1.8068546} ,
    'trackNumberOfPixelHits'  :  {'mean': 4.205281217365359, 'std': 0.9478835493717127} ,
    'trackNumberOfSCTHits'  :  {'mean': 8.432492913620766, 'std': 1.3268142597502794} ,
    'trackNumberOfPixelDeadSensors'  :  {'mean': 0.08175443831120394, 'std': 0.28097955502041233} ,
    'trackNumberOfSCTDeadSensors'  :  {'mean': 0.09279427122184096, 'std': 0.412992406261207} ,
    'trackNumberOfInnermostPixelLayerHits'  :  {'mean': 0.9870207369834403, 'std': 0.36292256189756517} ,
    'trackNumberOfNextToInnermostPixelLayerHits'  :  {'mean': 1.0522154259286887, 'std': 0.35834182262430175} ,
    'trackExpectInnermostPixelLayerHit'  :  {'mean': 0.9749365955542294, 'std': 0.15631772197469573} ,
    'trackExpectNextToInnermostPixelLayerHit'  :  {'mean': 0.9684469640459495, 'std': 0.17480687022005978} ,
    'trackNumberOfTRTHits'  :  {'mean': 23.086379233179173, 'std': 13.239640400661633} ,
    'trackNumberOfTRTOutliers'  :  {'mean': 1.3385051469491274, 'std': 6.073115255017552} ,
    'trackChiSquared'  :  {'mean': 35.072063, 'std': 17.583935} ,
    'trackNumberDOF'  :  {'mean': 34.92943458153066, 'std': 12.928489800018584} ,
    'trackD0'  :  {'mean': -0.011262955, 'std': 1.268558} ,
    'trackZ0'  :  {'mean': -0.49036446, 'std': 43.61} ,
    'truthPartE'  :  {'mean': 215.88905, 'std': 411.5805}
}

# --- K_NN graph dataclass ----------------#
class MLPionsDataset_KNN(Dataset):
    def __init__(self, filename, k_val, cluster_var, nstart, nstop, cluster_var_dict_musig=cluster_var_dict_musig ):
        
        self.k = k_val
        self.f = uproot.open(filename)
        self.cluster_var = cluster_var
        
        self.ev_tree =  self.f['EventTree;1']
        self.cell_geo_tree = self.f['CellGeo;1']
        
        self.nstart = nstart
        self.nstop = nstop
        
        if(self.nstop - self.nstart > self.ev_tree.num_entries) : 
            self.nstop = self.nstart + self.ev_tree.num_entries

        self.n_events = self.nstop - self.nstart

        self.cluster_var_dict_musig = cluster_var_dict_musig

#         for ivar in self.cluster_var : 
#             var = np.concatenate(self.ev_tree[ivar].array(library='np'))
#             self.cluster_var_dict_musig[ivar] =  {'mean' : var.mean(), 'std' : var.std()}
        
        self.cell_positions, self.id_to_position,\
        self.id_to_deta, self.id_to_dphi = self.get_all_cells()
        
        self.all_graphs = []


        for i in tqdm(range( self.nstart, self.nstop, 1 )):
            event_dict = self.get_single_event(i)
            #if(event_dict['truth_E'].item() >=0. ) : 
            self.all_graphs.append(event_dict)
    
#         self.n_eff = len(self.all_graphs)
    
    def get_single_event(self,event_idx):

        # ------- building the cluster graph ---------- #
        
        cluster_cell_ID = self.ev_tree['cluster_cell_ID'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        cluster_cell_E = self.ev_tree['cluster_cell_E'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        cluster_E = self.ev_tree['cluster_E'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        
        n_clusters = len(cluster_cell_ID)
        ev_flag = 1
        
        # -- empty dict to store all the node features --- #
        node_var_dict = { var : [] for var in self.cluster_var}
        
        nTrack = self.ev_tree['nTrack'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        
        if(n_clusters == 0) : 
            #print('Empty cluster event')
            ev_flag = -1
            return {'gr' : [dgl.rand_graph(2,1)], 'truth_E' : torch.tensor([-1.]), 'truth_eta' : torch.tensor([-99.])}#, ev_flag
        
        if(nTrack > 1) : 
            #print('Empty cluster event')
            ev_flag = -1
            return {'gr' : [dgl.rand_graph(2,1)], 'truth_E' : torch.tensor([-1.]), 'truth_eta' : torch.tensor([-99.])}#, ev_flag
        # ----------- determining track_xyz ----------------- #
        track_layer_eta_phi =\
        {
          lname : np.array(
                    [
                     self.ev_tree['trackEta_'+lname].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0],
                     self.ev_tree['trackPhi_'+lname].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
                    ]
            )#.reshape(2,) 
    
              for lname in layer_name
        }
        
        ftrack_lep =\
        {
          lname : value.reshape(2,)  for lname, value in track_layer_eta_phi.items()\
            if len(value) > 0 and  value[0] > -20                    
            
        }
        
        n_trhits = len(ftrack_lep)
        
        if(n_trhits == 0) : 
            ev_flag = -1
            return {'gr' : [dgl.rand_graph(2,1)], 'truth_E' : torch.tensor([-1.]), 'truth_eta' : torch.tensor([-99.])}#, ev_flag
            
        track_xyz = []
    
        for ft in ftrack_lep : 
            region = layer_name[ft][0]
            rperp = layer_name[ft][1]
            eta   = ftrack_lep[ft][0]
            phi   = ftrack_lep[ft][1]

            theta = 2*np.arctan( np.exp(-eta) )
            
            if(region == 1) : 
                rperp = rperp/np.sin(theta)

            trk_x = rperp*np.cos(phi)
            trk_y = rperp*np.sin(phi)
            trk_z = rperp/np.tan(theta)

            track_xyz.append( torch.tensor([trk_x, trk_y, trk_z]) )
            
        
        track_xyz = torch.vstack(track_xyz) 
        
        trackP = self.ev_tree['trackP'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0]
        
        # ------------------------------------------------------ #
        cluster_var_dict = {}

        for ivar in self.cluster_var : 
            cluster_var_dict[ivar] =\
             ( torch.tensor(self.ev_tree[ivar].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0])
                - self.cluster_var_dict_musig[ivar]['mean']
                )/self.cluster_var_dict_musig[ivar]['std']
               

        graph_list, clus_var  = [], []
        cell_E, cluster_cell_pos, cluster_cell_deta, cluster_cell_dphi = [], [], [], []
        # ---- loop over clusters ---- #
        for ic in range(n_clusters) : 
            
            if(cluster_E[ic] > 0.5) : 
                #cell_E.append( cluster_cell_E[ic] )
                cell_E += cluster_cell_E[ic].tolist()
                cell_idx = np.array(cluster_cell_ID[ic])
                n_cell = len(cell_idx)

                #print([ torch.tensor(self.id_to_deta[x]) for x in cell_idx])

                cluster_cell_pos.append( torch.cat([ torch.tensor([self.id_to_position[x]]) for x in cell_idx]) )
                cluster_cell_deta.append( torch.cat([ torch.tensor([self.id_to_deta[x]]) for x in cell_idx]) )
                cluster_cell_dphi.append( torch.cat([ torch.tensor([self.id_to_dphi[x]]) for x in cell_idx]) )

                for ivar in self.cluster_var : 

                    if(ivar.find('track') == -1) : 

                        node_var_dict[ivar].append(  
                                        torch.repeat_interleave(
                                            torch.reshape( cluster_var_dict[ivar][ic] , (1, 1) ), n_cell, dim=0
                                        )  # repeat_interleave
                                       ) # append
                    else : 

                        node_var_dict[ivar].append(  
                                        torch.repeat_interleave(
                                            torch.reshape( cluster_var_dict[ivar] , (1, 1) ), n_cell, dim=0
                                        )  # repeat_interleave
                                       ) # append

                    
            
        # -- check if any cluster survives ---- #
        if( len(cluster_cell_deta) == 0 ) :
            ev_flag = -1
            return {'gr' : [dgl.rand_graph(2,1)], 'truth_E' : torch.tensor([-1.])}#, ev_flag
                    
            
            
            
            
        # --- filling for the track nodes --- #
        for ivar in self.cluster_var : 
                
                if(ivar.find('track') == -1) : 
                    
                    node_var_dict[ivar].append(  
                                    torch.repeat_interleave(
                                        torch.reshape( cluster_var_dict[ivar].mean() , (1, 1) ), n_trhits, dim=0
                                    )  # repeat_interleave
                                   ) # append
                else : 
                    
                    node_var_dict[ivar].append(  
                                    torch.repeat_interleave(
                                        torch.reshape( cluster_var_dict[ivar] , (1, 1) ), n_trhits, dim=0
                                    )  # repeat_interleave
                                   ) # append
            
        #cell_E.append( torch.tensor([ trackP for _ in range(n_trhits) ]) )
        cell_E = cell_E + [ trackP[0] for _ in range(n_trhits) ]
        cluster_cell_pos.append( track_xyz)
        cluster_cell_deta.append( torch.tensor([ torch.cat(cluster_cell_deta, dim=0).mean() for _ in range(n_trhits) ])  )
        cluster_cell_dphi.append(  torch.tensor([ torch.cat(cluster_cell_dphi, dim=0).mean() for _ in range(n_trhits) ])  )           

        #print(cell_E)
        cell_E = torch.tensor(cell_E)
        #print('cell_E shape : ', cell_E.shape)
        cluster_cell_pos = torch.cat(cluster_cell_pos, dim=0)
        cluster_cell_deta = torch.cat(cluster_cell_deta, dim=0)
        cluster_cell_dphi = torch.cat(cluster_cell_dphi, dim=0)
        
        n_part = len(cluster_cell_pos)
            
        if(n_part < self.k) : 
            knn_g = dgl.knn_graph(cluster_cell_pos , n_part)
        else : 
            knn_g = dgl.knn_graph(cluster_cell_pos , self.k)

        knn_g.ndata['x'] = cluster_cell_pos.float()

            

        clus_var.append( torch.reshape(cluster_cell_deta, (cluster_cell_deta.shape[0], 1) ) )
        clus_var.append( torch.reshape(cluster_cell_dphi, (cluster_cell_dphi.shape[0], 1) ) )

        clus_var.insert(0, torch.reshape(cell_E, (cell_E.shape[0], 1) ) )
            
        for var in self.cluster_var : 
            
            #print(var, torch.cat(node_var_dict[var], dim=0).shape )
            clus_var.append(torch.cat(node_var_dict[var], dim=0))
                

        knn_g.ndata['en'] = torch.stack(clus_var, dim=2).float()
        
        

        #graph_list.append(knn_g)
        # -------- #
        
        cluster_energy_truth =\
        self.ev_tree['truthPartE'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0] 

        truth_eta = self.ev_tree['truthPartEta'].array(entry_start=event_idx,entry_stop=event_idx+1,library='np')[0] 
        
        #cluster_energy_truth = (cluster_energy_truth - self.cluster_var_dict_musig['truthPartE']['mean'])/self.cluster_var_dict_musig['truthPartE']['std']
        # ---------------------------------------------------------------- #
        return {'gr' : knn_g, 'truth_E' : torch.tensor(cluster_energy_truth), 'truth_eta' : torch.tensor(truth_eta) }#, ev_flag
        
    
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
            #return self.n_eff 
            # else : 
            return self.n_events
        
    def __getitem__(self, idx):

            return self.all_graphs[idx]
            #return self.get_single_event(idx)


# -------------------------------------------------- #


def collate_graphs(event_list) : 
    
    n_batch = len(event_list)
    full_gr_list = []
    energy_list = []
    
    for ib in range(n_batch) : 
       
        #print(event_list[ib])
        gr_list = event_list[ib]['gr']
        tr_elist = event_list[ib]['truth_E']
        #energy_list.append(event_list[ib]['truth_E'])
        
        #for i in range(len(gr_list)) : 
        if( tr_elist.item() >=0. ) : 
            full_gr_list.append(gr_list)
            energy_list.append(tr_elist)
                  
    return dgl.batch(full_gr_list), torch.tensor(energy_list)

def collate_graphs_eta(event_list) : 
    
    n_batch = len(event_list)
    full_gr_list = []
    energy_list = []
    eta_list = []
    
    for ib in range(n_batch) : 

        #print(event_list[ib])
       
        #print(event_list[ib])
        gr_list = event_list[ib]['gr']
        tr_elist = event_list[ib]['truth_E']

        #tr_etal  = event_list[ib]['truth_eta']
        #energy_list.append(event_list[ib]['truth_E'])
        
        #for i in range(len(gr_list)) : 
        if( tr_elist.item() >=0. ) : 
            full_gr_list.append(gr_list)
            energy_list.append(tr_elist)
            eta_list.append(event_list[ib]['truth_eta'])
                  
    return dgl.batch(full_gr_list), torch.tensor(energy_list), torch.tensor(eta_list)
