import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

import h5py
import uproot#3 as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

torch.manual_seed(0)

import os, sys

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("--model_name", help="choose the model type", type=str)
parser.add_argument("--dev", help="choose the device node", type=str)
args = parser.parse_args()

#model_name = args.model_name
device = args.dev


#os.environ["CUDA_VISIBLE_DEVICES"]=device#"0"
cuda_device = torch.device('cuda:'+device if torch.cuda.is_available() else 'cpu' )

print('cuda_device : ', cuda_device)

from modules.ML4Pions_Dataset import  MLPionsDataset_Set, collate_graphs

from modules.set_transformer import SetTransformer

cluster_var = ['cluster_EM_PROBABILITY', 'cluster_HAD_WEIGHT', 'cluster_OOC_WEIGHT',
               'cluster_DM_WEIGHT', 'cluster_CENTER_MAG', 'cluster_FIRST_ENG_DENS', 
               'cluster_CENTER_LAMBDA', 'cluster_ISOLATION'
              ]

track_var = ['trackPt',
             'trackP',
             'trackMass',
             'trackEta',
             'trackPhi',
             'trackNumberOfPixelHits',
             'trackNumberOfSCTHits',
             'trackNumberOfPixelDeadSensors',
             'trackNumberOfSCTDeadSensors',
#              'trackNumberOfPixelSharedHits',
#              'trackNumberOfSCTSharedHits',
#              'trackNumberOfPixelHoles',
#              'trackNumberOfSCTHoles',
             'trackNumberOfInnermostPixelLayerHits',
             'trackNumberOfNextToInnermostPixelLayerHits',
             'trackExpectInnermostPixelLayerHit',
             'trackExpectNextToInnermostPixelLayerHit',
             'trackNumberOfTRTHits',
             'trackNumberOfTRTOutliers',
             'trackChiSquared',
             'trackNumberDOF',
             'trackD0',
             'trackZ0'
            ]


file_name_test = 'samples/test_dnn_sanmay.h5'

n_test = 100000
n_slice = 1000

test_data = torch.utils.data.ConcatDataset([
            MLPionsDataset_Set(filename=file_name_test, cluster_var=cluster_var, track_var=track_var, nstart=i, nstop=i+n_slice)\
            for i in range( 0,  n_test, n_slice )            
    ])


test_loader = DataLoader(test_data, batch_size=10, shuffle=False,collate_fn=collate_graphs, num_workers=0)

#model = SetTransformer(n_enc = 3, d_model = 27, n_heads = 4, d_head = 4, d_ff = 20, n_layers = 4, k = 2)
model = SetTransformer(n_enc = 4, d_model = 27, n_heads = 5, d_head = 4, d_ff = 20, n_layers = 5, k = 2)
#model_name = 'model_SetTransformer_MSE10.pt'
model_name = 'model_SetTransformer_PTNL.pt'

model.to(cuda_device)

model.load_state_dict(torch.load(model_name))

param_numb = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total parameters : ', param_numb)

model.eval()

pred_energy, target_energy, truth_eta = [], [], []
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

# pred_energy = (pred_energy )*465.8594184884767 + 324.16019827205093
# target_energy = (target_energy )*465.8594184884767 + 324.16019827205093
# pred_energy = (10 ** pred_energy )
# target_energy = (10 ** target_energy)


pred_energy = pred_energy[ np.where(target_energy!=0) ]
target_energy = target_energy[ np.where(target_energy!=0) ]


resp = (pred_energy)/target_energy

print('mean R : ', np.median( resp ) )
print('std R : ', resp.std() )

tot_pred_energy, tot_target_energy = np.array(tot_pred_energy), np.array(tot_target_energy)
tot_pred_energy = tot_pred_energy[np.where(tot_target_energy!=0)]
tot_target_energy = tot_target_energy[np.where(tot_target_energy!=0)]

resp_tot = tot_pred_energy/tot_target_energy

print('total mean R : ', np.median( resp_tot ) )
print('total std R : ', resp_tot.std() )

out_name = 'PredictionFile_SetTransformer_PTNL.h5'

hf = h5py.File(out_name, 'w')

hf.create_dataset('pred_energy', data=pred_energy, compression='lzf')
hf.create_dataset('target_energy', data=target_energy, compression='lzf')
#hf.create_dataset('truth_eta', data=truth_eta, compression='lzf')
hf.create_dataset('tot_pred_energy', data=tot_pred_energy, compression='lzf')
hf.create_dataset('tot_target_energy', data=tot_target_energy, compression='lzf')

hf.close()
