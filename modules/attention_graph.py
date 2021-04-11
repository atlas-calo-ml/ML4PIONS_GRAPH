import dgl
import dgl.function as fn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import math

from modules.mlp import build_mlp

# def src_dot_dst(src_field, dst_field, out_field):
#     def func(edges):
#         return {out_field: torch.matmul(edges.src[src_field], edges.dst[dst_field].transpose(-2, -1)).sum(-1, keepdim=True)}
#     return func

# def scaled_exp(field, scale_constant):
#     def func(edges):
#         # clamp for softmax numerical stability
#         #return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
#         return {field: F.softmax((edges.data[field] / scale_constant), dim=-2)}

#     return func

"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, layer_indx, input_names='node_attention_head'):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.input_names = input_names

        self.K_st = 'K_h' + layer_indx
        self.Q_st = 'Q_h' + layer_indx
        self.score_st = 'score' + layer_indx
        self.V_st = 'V_h' + layer_indx
        self.wV_st = 'wV' + layer_indx
        self.z_st = 'z' + layer_indx
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    
    def propagate_attention(self, g):
        # Compute attention score
        
        g.apply_edges(self.src_dot_dst(self.K_st, self.Q_st, self.score_st)) #, edges)
        g.apply_edges(self.scaled_exp(self.score_st, np.sqrt(self.out_dim)))

        
        # Send weighted values to target nodes
        eids = g.edges()

        g.send_and_recv(eids, fn.u_mul_e(self.V_st, self.score_st, self.V_st), fn.sum(self.V_st, self.wV_st))
        g.send_and_recv(eids, fn.copy_edge(self.score_st, self.score_st), fn.sum(self.score_st, self.z_st))


    def src_dot_dst(self, src_field, dst_field, out_field):
        def func(edges):
            return {out_field: torch.matmul(edges.src[src_field], edges.dst[dst_field].transpose(-2, -1)).sum(-1, keepdim=True)}
        return func

    def scaled_exp(self, field, scale_constant):
        def func(edges):
            # clamp for softmax numerical stability
            return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
            #return {field: F.softmax((edges.data[field] / scale_constant), dim=-2)}
        return func


    def forward(self, g):
        
        if(self.input_names == 'node_attention_head') : 
            h = g.ndata['node_attention_head']
        else : 
            h =  g.ndata['en']

        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata[self.Q_st] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata[self.K_st] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata[self.V_st] = V_h.view(-1, self.num_heads, self.out_dim)
        
        g = dgl.add_self_loop(g)
        self.propagate_attention(g)
        
        head_out = g.ndata[self.wV_st]/g.ndata[self.z_st]
        
        g = dgl.remove_self_loop(g)

        g.ndata['node_attention_head'] = torch.mean(head_out, dim=1)
        
        return g

# -------------------------- deepset layer with attention ------------------------------- #
class DeepSetLayer(nn.Module):
    def __init__(self,in_dim, out_dim, num_heads, layer_indx, use_bias = True,\
                                input_names='node_attention_head',\
                                            output_name='node_attention_head',apply_activation=True):
        super(DeepSetLayer, self).__init__()

        self.input_names = input_names
        self.output_name = output_name

        self.in_dim = in_dim
        self.out_dim = out_dim
    
        self.layer1 = build_mlp(inputsize  = out_dim,\
                                  outputsize = out_dim,\
                                  features = [64, 128, 32]
                                )

        self.layer2 = build_mlp(inputsize  = out_dim,\
                                  outputsize = out_dim,\
                                  features = [64, 128, 32]
                                )

        
        self.attention = MultiHeadAttentionLayer(in_dim = in_dim, out_dim = out_dim, num_heads = num_heads, use_bias = use_bias, 
                                   input_names = input_names, layer_indx=layer_indx
                                  )

        self.apply_activation = apply_activation
        self.activation = nn.LeakyReLU(-1.0)

        #self.layer_norm = nn.LayerNorm(out_dim)


    def forward(self, g):
        
        # ---- apply the node attention -- #
        g = self.attention(g)

        node_data = g.ndata['node_attention_head']

        mean_node_inputs = dgl.broadcast_nodes(g,dgl.mean_nodes(g,'node_attention_head'))

    

        x = self.layer1(node_data)+self.layer2(node_data-mean_node_inputs)
        

        #x = self.layer_norm(x)

        if self.apply_activation:
            g.ndata[self.output_name] = self.activation(x)
        else : 
            g.ndata[self.output_name] = x
        
        
        return g



class Graph_Attention_Model(nn.Module):
    def __init__(self, num_heads, feature_dims,  use_bias = True,\
                                  input_names='node_attention_head',\
                                            output_name='node_attention_head',apply_activation=True):
        super(Graph_Attention_Model, self).__init__()

        self.n_layers = len(feature_dims)-1

        self.initial_feat = input_names 
        
        self.layer_list = []

        self.layer_list.append( 

                        DeepSetLayer(in_dim = len(self.initial_feat) + 3, out_dim = feature_dims[0], num_heads = num_heads, use_bias = True, 
                                   input_names = self.initial_feat, apply_activation = apply_activation, layer_indx='_enc0'
                                  )
         )


        for i_l in range(self.n_layers) : 

            self.layer_list.append(

                        DeepSetLayer(in_dim = feature_dims[i_l], out_dim = feature_dims[i_l+1],\
                                     num_heads = feature_dims[i_l], use_bias = True, apply_activation = apply_activation,
                                     layer_indx='_enc' + str(i_l+1)
                                     )
                )

        self.layers = nn.ModuleList(self.layer_list)

        self.latent_project = build_mlp(inputsize = sum(feature_dims),\
                                        outputsize = 1,\
                                        features = [64, 128, 32]
                                        )


    def forward(self, g):
        
        with g.local_scope() : 
            out_energy = []

            graph_list = dgl.unbatch(g)

            for ig in graph_list : 
                e_array = []
                for il in range(self.n_layers+1) : 
        
                    ig = self.layer_list[il](ig)                
                    e_array.append( dgl.mean_nodes(ig, feat='node_attention_head') )
                
                e_array = self.latent_project(torch.cat(e_array, dim=1))
                
                out_energy.append(e_array[0])

            return torch.cat(out_energy, dim=0)



