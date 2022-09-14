import dgl
import dgl.function as fn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import math

from dgl.nn import SetTransformerEncoder, SetTransformerDecoder

from modules.mlp import build_mlp


# ---- encoder layer ---------- #

class EncoderLayer(nn.Module):
    def __init__(self,d_model, n_heads, d_head, d_ff, n_layers=1, block_type='sab', m=None, dropouth=0.0, dropouta=0.0):
        super(EncoderLayer, self).__init__()

    
        self.enc = SetTransformerEncoder(d_model = d_model, n_heads = n_heads, d_head = d_head, d_ff = d_ff,
                                         n_layers=n_layers, block_type=block_type, m=m, dropouth=dropouth, dropouta=dropouta)

        


    def forward(self, g):
        
        # ---- apply the node attention -- #
        n_data = g.ndata['en']
        
        n_data_o = self.enc(g, feat = n_data) + n_data

        g.ndata['en'] = n_data_o

        
        return g


# ---------------- decoder layer ------------- #

class DecoderLayer(nn.Module):
    def __init__(self,d_model, n_heads, d_head, d_ff, n_layers, k, dropouth=0.0, dropouta=0.0):
        super(DecoderLayer, self).__init__()

    
        self.dec = SetTransformerDecoder(d_model = d_model, num_heads = n_heads, d_head = d_head, d_ff = d_ff,
                                         n_layers=n_layers, k = k, dropouth=dropouth, dropouta=dropouta)

        


    def forward(self, g):
        
        # ---- apply the node attention -- #
        n_data = g.ndata['en']
        
        n_data_o = self.dec(g, feat = n_data)

        
        return n_data_o

# --- set transformer model ----------------- #

class SetTransformer(nn.Module):
    def __init__(self,n_enc, d_model, n_heads, d_head, d_ff, k, n_layers=1, 
                 block_type='sab', m=None, dropouth=0.0, dropouta=0.0):
        super(SetTransformer, self).__init__()

    
        self.n_enc = n_enc
        self.enc_list = []
        
        for i_l in range(n_enc) : 
            self.enc_list.append( 
                       EncoderLayer(d_model = d_model, n_heads = n_heads, d_head = d_head, d_ff = d_ff,
                                             n_layers=n_layers, block_type=block_type, m=m, dropouth=dropouth,
                                             dropouta=dropouta)
                                )
            
        
        self.enc_list = nn.ModuleList(self.enc_list)
        
        self.dec = DecoderLayer(d_model = d_model, n_heads = n_heads, d_head = d_head, d_ff = d_ff,
                                         n_layers=n_layers, k = k, dropouth=dropouth, dropouta=dropouta)
        

        self.mlp = build_mlp(inputsize = d_model * k, outputsize = 1,\
                                         features = [40, 20, 10, 5, 3]
                                        )

    def forward(self, g):
        
        # ---- apply the node attention -- #
        for il in range(self.n_enc) : 
            
            g = self.enc_list[il](g)
        
        
        n_data_o = self.dec(g) 

        n_data_o = self.mlp(n_data_o)

        
        return n_data_o
