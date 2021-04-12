import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

import h5py

torch.manual_seed(0)

import os, sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="choose the model type", type=str)
args = parser.parse_args()

model_name = args.model_name

os.environ["CUDA_VISIBLE_DEVICES"]="0"
cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

print('cuda_device : ', cuda_device)

from modules.ML4Pions_Dataset import  MLPionsDataset_KNN, collate_graphs

from modules.dynamic_graph import Dynamic_Graph_Model
from modules.attention_graph import Graph_Attention_Model

cluster_var = ['cluster_EM_PROBABILITY', 'cluster_HAD_WEIGHT', 'cluster_OOC_WEIGHT',
               'cluster_DM_WEIGHT', 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 
               'cluster_CENTER_LAMBDA', 'cluster_ISOLATION',  
              ]

file_name_test = 'samples/ml4pions_test.root'

test_data = MLPionsDataset_KNN(filename=file_name_test, k_val=5, cluster_var=cluster_var, num_ev=-1)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,collate_fn=collate_graphs, num_workers=0)

#model = Dynamic_Graph_Model(feature_dims_x = [8, 9, 7, 5], feature_dims_en = [4, 5, 6, 8])
if(model_name == 'edgeconv') : 
    model = Dynamic_Graph_Model(feature_dims_x = [8, 9, 7, 5], feature_dims_en = [4, 5, 6, 8])
    model_name = 'model_DynamicGraph.pt'
    out_name = 'PredictionFile_DynamicGraph.h5'
else : 
    model = Graph_Attention_Model(num_heads = 5, feature_dims = [10, 15, 12, 8], input_names=cluster_var)
    model_name = 'model_AttentionGraph.pt'
    out_name = 'PredictionFile_AttentionGraph.h5'
#model = nn.DataParallel(model)
model.to(cuda_device)

model.load_state_dict(torch.load(model_name))

param_numb = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total parameters : ', param_numb)

model.eval()

pred_energy, target_energy = [], []
tot_pred_energy, tot_target_energy = [], []

with tqdm(test_loader, ascii=True) as tq:
    for gr, truth_calib in tq:
        
        gr, truth_calib = gr.to(cuda_device), truth_calib.to(cuda_device)
        
        pred_calib = model( gr )

        pred_energy.append( pred_calib.cpu().detach().numpy() )
        target_energy.append( truth_calib.cpu().detach().numpy() )
        
        tot_pred_energy.append( np.sum(pred_calib.cpu().detach().numpy()) )
        tot_target_energy.append( np.sum(truth_calib.cpu().detach().numpy()) )
        
        
        del gr; del truth_calib; del pred_calib;


pred_energy, target_energy = np.concatenate( pred_energy ), np.concatenate( target_energy )

pred_energy = pred_energy[ np.where(target_energy!=0) ]
target_energy = target_energy[ np.where(target_energy!=0) ]

resp = (pred_energy)/target_energy

print('mean R : ', resp.mean() )
print('std R : ', resp.std() )

tot_pred_energy, tot_target_energy = np.array(tot_pred_energy), np.array(tot_target_energy)
tot_pred_energy = tot_pred_energy[np.where(tot_target_energy!=0)]
tot_target_energy = tot_target_energy[np.where(tot_target_energy!=0)]

resp_tot = tot_pred_energy/tot_target_energy

print('total mean R : ', resp_tot.mean() )
print('total std R : ', resp_tot.std() )


hf = h5py.File(out_name, 'w')

hf.create_dataset('pred_energy', data=pred_energy, compression='lzf')
hf.create_dataset('target_energy', data=target_energy, compression='lzf')
hf.create_dataset('tot_pred_energy', data=tot_pred_energy, compression='lzf')
hf.create_dataset('tot_target_energy', data=tot_target_energy, compression='lzf')

hf.close()
