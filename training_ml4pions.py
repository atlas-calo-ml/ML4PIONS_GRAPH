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

file_name_train = 'samples/ml4pions_training.root'
file_name_valid = 'samples/ml4pions_validation.root'

train_data = MLPionsDataset_KNN(filename=file_name_train, k_val=5, cluster_var=cluster_var, num_ev=-1)
valid_data = MLPionsDataset_KNN(filename=file_name_valid, k_val=5, cluster_var=cluster_var, num_ev=-1)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True,collate_fn=collate_graphs, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=5, shuffle=False,collate_fn=collate_graphs, num_workers=0)

if(model_name == 'edgeconv') : 
    model = Dynamic_Graph_Model(feature_dims_x = [8, 9, 7, 5], feature_dims_en = [4, 5, 6, 8])
    model_name = 'model_DynamicGraph.pt'
else : 
    model = Graph_Attention_Model(num_heads = 5, feature_dims = [10, 15, 12, 8], input_names=cluster_var)
    model_name = 'model_AttentionGraph.pt'
#model = nn.DataParallel(model)
model.to(cuda_device)

opt = optim.Adam(model.parameters(), lr=1e-3)

# ---------------- Make the training loop ----------------- #

train_loss_v, valid_loss_v = [], []


# number of epochs to train the model
n_epochs = 100

valid_loss_min = np.Inf # track change in validation loss

loss_fn = nn.MSELoss()
loss_fn.to(cuda_device)

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
            
            loss = loss_fn(truth_calib, pred_calib)
            
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
            
            loss = loss_fn(truth_calib, pred_calib)
            
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
