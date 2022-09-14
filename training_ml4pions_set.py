import dgl
import dgl.function as fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import math

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

file_name_train = 'samples/train_dnn_sanmay.h5'
file_name_valid = 'samples/val_dnn_sanmay.h5'

n_train, n_valid = 300000, 100000
n_slice = 1000

# train_data = MLPionsDataset_KNN(filename=file_name_train, k_val=5, cluster_var=cluster_var+track_var, nstart=0, nstop=1000)
# valid_data = MLPionsDataset_KNN(filename=file_name_valid, k_val=5, cluster_var=cluster_var+track_var, nstart=0, nstop=1000)

train_data = torch.utils.data.ConcatDataset([
            MLPionsDataset_Set(filename=file_name_train, cluster_var=cluster_var, track_var=track_var, nstart=i, nstop=i+n_slice)\
            for i in range( 0, n_train, n_slice )            
    ])

valid_data = torch.utils.data.ConcatDataset([
            MLPionsDataset_Set(filename=file_name_valid, cluster_var=cluster_var, track_var=track_var, nstart=i, nstop=i+n_slice)\
            for i in range( 0, n_valid, n_slice ) 
    ])

train_loader = DataLoader(train_data, batch_size=100, shuffle=True,collate_fn=collate_graphs, num_workers=0)
valid_loader = DataLoader(valid_data, batch_size=100, shuffle=False,collate_fn=collate_graphs, num_workers=0)

model = SetTransformer(n_enc = 4, d_model = 27, n_heads = 5, d_head = 4, d_ff = 20, n_layers = 5, k = 2)

model_name = 'model_SetTransformer_PTNL.pt'

model.to(cuda_device)

opt = optim.AdamW(model.parameters(), lr=1e-3)

# ---------------- Make the training loop ----------------- #

train_loss_v, valid_loss_v = [], []


# number of epochs to train the model
n_epochs = 100

valid_loss_min = np.Inf # track change in validation loss

#loss_fn = nn.MSELoss()

def loss_fn(pred, tar) : 

    z =  torch.abs( (pred - tar) )
    
     
    return torch.sum(z) 

#loss_fn.to(cuda_device)

for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    #scheduler.step()
    model.train() ## --- set the model to train mode -- ##
    with tqdm(train_loader, ascii=True) as tq:
        for gr, truth_calib in tq:
            
            gr, truth_calib = gr.to(cuda_device), truth_calib.to(cuda_device)
            
            opt.zero_grad()
            
            pred_calib = model( gr )
            
            loss = loss_fn( pred_calib, truth_calib)
            
            loss.backward()
            #loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            opt.step()

            # update training loss
            train_loss += loss.item()
            
            del gr; del truth_calib; del pred_calib;
            
            
    #####################    
    # validate the model #
    ######################
    model.eval()
    with tqdm(valid_loader, ascii=True) as tq:
        for gr, truth_calib in tq:
            
            gr, truth_calib = gr.to(cuda_device), truth_calib.to(cuda_device)
            
            pred_calib = model( gr )
            
            loss = loss_fn( pred_calib, truth_calib)
            
            valid_loss += loss.item()
            
            del gr; del truth_calib; del pred_calib;
            
            
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    train_loss_v.append(train_loss) 
    valid_loss_v.append(valid_loss)
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), model_name)
        valid_loss_min = valid_loss
            
# ---- end of script ------ # 

train_loss_v, valid_loss_v = np.array(train_loss_v), np.array(valid_loss_v)

hf = h5py.File('SetTransformer_Loss_PTNL.h5', 'w')

hf.create_dataset('train_loss', data=train_loss_v, compression='lzf')
hf.create_dataset('valid_loss', data=valid_loss_v, compression='lzf')

hf.close()
