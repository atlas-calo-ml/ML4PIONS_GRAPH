import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F_n

import os, sys


import dgl
from dgl import backend as F
import dgl.function as fn
from dgl.nn.pytorch import KNNGraph

from modules.mlp import build_mlp

# ---------------- The EdgeConv function ---------------------- #
class EdgeConv(nn.Module):
    
    def __init__(self,
                 in_feat_x,
                 out_feat_x,
                 in_feat_en,
                 out_feat_en,
                 batch_norm=False, 
                 k_val=5):
        super(EdgeConv, self).__init__()
        
        self.batch_norm = batch_norm
        #self.nng = KNNGraph(k_val)

        self.k = k_val
        
        
        self.theta_en = build_mlp(inputsize  = in_feat_en,\
                                  outputsize = out_feat_en,\
                                  features = [64, 128, 32],\
                                  add_batch_norm = batch_norm
                                  )
        
        self.phi_en   = build_mlp(inputsize  = in_feat_en,\
                                  outputsize = out_feat_en,\
                                  features = [64, 128, 32],\
                                  add_batch_norm = batch_norm
                                  )

        self.theta = nn.Linear(in_feat_x, out_feat_x)
        self.phi = nn.Linear(in_feat_x, out_feat_x)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat_x)

    def message(self, edges):
        """The message computation function.
        """
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        
        theta_en = self.theta_en(edges.dst['en'] - edges.src['en'])
        phi_en = self.phi_en(edges.src['en'])
        
        return {'edge_x': theta_x + phi_x, 
                'edge_en' :  phi_en + theta_en 
                }

    def forward(self, g):
        """Forward computation
        """
        if not self.batch_norm:
            
            g.apply_edges(self.message)
            g.update_all(fn.copy_e('edge_x', 'edge_x'), fn.max('edge_x', 'x'))
            g.update_all(fn.copy_e('edge_en', 'edge_en'), fn.mean('edge_en', 'en'))
        else:
            g.apply_edges(self.message)

            g.edata['edge_x'] = self.bn(g.edata['edge_x'])

            g.update_all(fn.copy_e('edge_x', 'edge_x'), fn.max('edge_x', 'x'))

            g.update_all(fn.copy_e('edge_en', 'edge_en'), fn.mean('edge_en', 'en'))

        #print('New X shape : ', g.ndata['x'].shape)

        #g_new = self.nng( g.ndata['x'] )

        n_part = len(g.ndata['x'])

        if(n_part < self.k) : 
                g_new = dgl.knn_graph(g.ndata['x'] , n_part)
        else : 
                g_new = dgl.knn_graph(g.ndata['x'] , self.k)
       
        dev = g.device.type
        g_new = g_new.to(dev) 
        g_new.ndata['x'] = g.ndata['x']
        g_new.ndata['en'] = g.ndata['en']

        dev = g.device.type        
        del g
        return g_new


# --- the full dynamic edgeconv model ---- #
class Dynamic_Graph_Model(nn.Module):
    def __init__(self, feature_dims_x, feature_dims_en):
        super(Dynamic_Graph_Model, self).__init__()

        self.n_layers = len(feature_dims_x)-1
        
        self.layer_list = nn.ModuleList()

        self.layer_list.append( 

                        EdgeConv(in_feat_x = 3, out_feat_x = feature_dims_x[0], in_feat_en = 11,\
                                 out_feat_en = feature_dims_en[0])
         )


        for i_l in range(self.n_layers) : 

            self.layer_list.append(

                        EdgeConv(in_feat_x = feature_dims_x[i_l], out_feat_x = feature_dims_x[i_l+1],\
                                 in_feat_en = feature_dims_en[i_l], out_feat_en = feature_dims_en[i_l+1])
                )
            

        self.latent_project = build_mlp(inputsize = sum(feature_dims_en),\
                                        outputsize = 1,\
                                        features = [64, 128, 32]
                                        )
                 
    # -- the forward function -- #
    def forward(self, g):
        
        with g.local_scope() : 
            out_energy = []

            graph_list = dgl.unbatch(g)

            for ig in graph_list : 
                e_array = []
                for il in range(self.n_layers+1) : 
        
                    ig = self.layer_list[il](ig)                
                    e_array.append( dgl.mean_nodes(ig, feat='en')[0] )
                    
                e_array = self.latent_project(torch.cat(e_array, dim=1))
                out_energy.append(e_array[0])
                
            return  torch.cat( out_energy ) 

