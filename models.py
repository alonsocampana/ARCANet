import pandas as pd
import torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import numpy as np
from torch_geometric import nn as gnn
import torchmetrics
import tqdm
from functools import lru_cache
import torch_geometric

class MultiplicativeGaussianNoise(nn.Module):
    def __init__(self, var):
        super().__init__()
        self.var = var
    def forward(self, x):
        if self.training:
            factor = x.new_ones(x.shape)
            factor += torch.randn_like(factor)*self.var
            return x * factor
        else:
            return x

class GatedResidualSequential(nn.Sequential):
    def __init__(self, modulelist,
                 use_layernorm = False,
                 *args, **kwargs):
        super().__init__(*modulelist, *args, **kwargs)
        self.use_layernorm = use_layernorm
        if isinstance(self[0], nn.Linear):
            self.input_shape = self[0].weight.shape[1]
        else:
            self.input_shape = self[1].weight.shape[1]
        self.layernorm = nn.BatchNorm1d(self.input_shape)
        self.gate = torch.nn.Parameter(torch.zeros(1))
    def forward(self, input):
        alpha = torch.sigmoid(self.gate)
        x = torch.clone(input)
        for i in range(len(self) - 1):
            input = self[i](input)
        input = (alpha * x) + ((1-alpha) * input)
        if self.use_layernorm:
            if len(input.shape) == 3:
                input = input.transpose(1, 2)
            input = self.layernorm(input)
            if len(input.shape) == 3:
                input = input.transpose(2, 1)
        return input

class GatedResidualSequentialOld(nn.Sequential):
    def __init__(self, modulelist,
                 use_layernorm = False,
                 *args, **kwargs):
        super().__init__(*modulelist, *args, **kwargs)
        self.use_layernorm = use_layernorm
        if isinstance(self[0], nn.Linear):
            self.input_shape = self[0].weight.shape[1]
        else:
            self.input_shape = self[1].weight.shape[1]
        self.layernorm = nn.BatchNorm1d(self.input_shape)
    def forward(self, input):
        x = torch.clone(input)
        for i in range(len(self) - 1):
            input = self[i](input)
        input =  x + input
        if self.use_layernorm:
            if len(input.shape) == 3:
                input = input.transpose(1, 2)
            input = self.layernorm(input)
            if len(input.shape) == 3:
                input = input.transpose(2, 1)
        return input

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class GATRes(nn.Module):
    def __init__(
        
        self,
        init_dim = 9,
        hidden_dim = 128,
        edge_dim = 3,
        n_layers = 4,
        n_heads = 4,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.init_gat = gnn.GATv2Conv(init_dim, hidden_dim, edge_dim=edge_dim, heads=1)
        self.gats = nn.ModuleList([gnn.GATv2Conv(hidden_dim, hidden_dim, concat=False, edge_dim=edge_dim, heads=n_heads) for i in range(n_layers - 1)])
        self.gates = nn.Parameter(torch.zeros(n_layers - 1))
        self.noise = MultiplicativeGaussianNoise(0.1)
    def forward(self, x, edge_index,edge_weight = None):
        gates = torch.sigmoid(self.gates)
        x = self.init_gat(x, edge_index, edge_weight)
        for i in range(self.n_layers - 1):
            x = (1 - gates[i]) * self.gats[i](F.leaky_relu(x), edge_index, edge_weight) + ((gates[i]) * x)
        return x
class GATResOld(nn.Module):
    def __init__(
        
        self,
        init_dim = 9,
        hidden_dim = 128,
        edge_dim = 3,
        n_layers = 4,
        n_heads = 4,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.init_gat = gnn.GATv2Conv(init_dim, hidden_dim, edge_dim=edge_dim, heads=1)
        self.gats = nn.ModuleList([gnn.GATv2Conv(hidden_dim, hidden_dim, concat=False, edge_dim=edge_dim, heads=n_heads) for i in range(n_layers - 1)])
        self.gates = nn.Parameter(torch.zeros(n_layers - 1))
        self.noise = MultiplicativeGaussianNoise(0.1)
    def forward(self, x, edge_index,edge_weight = None):
        gates = torch.sigmoid(self.gates)
        x = self.init_gat(x, edge_index, edge_weight)
        for i in range(self.n_layers - 1):
            x = self.gats[i](F.leaky_relu(x), edge_index, edge_weight) + (x)
        return x

        
class ResNetGated(nn.Module):
    def __init__(self, init_dim, hidden_dim, layers, p_dropout):
        super().__init__()
        self.p_dropout = p_dropout
        assert layers > 1
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(init_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Dropout(p=p_dropout),
                             nn.Linear( hidden_dim, init_dim)) for i in range(layers)])
        self.gates = nn.Parameter(torch.Tensor(layers))
        self.layers.apply(init_weights)
    def forward(self, x):
        range_gates = torch.sigmoid(self.gates)
        for i, layer in enumerate(self.layers):
            x = (range_gates[i])*layer(F.relu(x)) + (1-range_gates[i])*x
        return x
    
class GeneEncoder(nn.Module):
    def __init__(self, init_dim,
                 bottleneck_dim,
                 p_dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(nn.Dropout(p=p_dropout),
                            nn.Linear(init_dim, bottleneck_dim))
        self.layers.apply(init_weights)
    def forward(self, x):
        return self.layers(x)
class AttnDropout(nn.Module):
    def __init__(self,
                 p_dropout = 0.1,
                 **kwargs):
        super().__init__()
        self.id = nn.Identity()
        self.bern = torch.distributions.bernoulli.Bernoulli(torch.Tensor([p_dropout]))
    def forward(self, x):
        x = self.id(x)
        if self.training:
            mask = self.bern.sample([x.shape[0]]).squeeze()
            x[mask.bool()] = float("-inf")
        return x
    
class ARCANetNew(nn.Module):
    def __init__(self,
                 p_dropout_attn = 0.3,
                 p_dropout_fc = 0.4,
                 dropout_genes = 0.3,
                 dropout_nodes = 0.2,
                 embed_dim=512,
                 n_pooling_heads = 4,
                 gat_heads = 2,
                 gat_layers = 4,
                 target_dim = 166,
                 fc_hidden = 2048,
                 activation = 0,
                 gaussian_noise = 0,
                 **kwargs):
        super().__init__()
        if activation == 0:
            activation = nn.Sigmoid
        elif activation == 1:
            activation = nn.ELU
        elif activation == 2:
            activation = nn.ReLU
        else:
            activation = nn.Tanh
        self.noise = MultiplicativeGaussianNoise(gaussian_noise)
        self.geneencoder = GeneEncoder(2089, embed_dim, p_dropout=dropout_genes)
        self.n_pooling_heads = n_pooling_heads
        self.p_dropout_fc = p_dropout_fc
        self.gat_encoder = GATRes(init_dim = 79,
                                  hidden_dim = embed_dim,
                                  n_heads=gat_heads,
                                  concat=False,
                                  edge_dim=10,
                                  n_layers = gat_layers)
        self.graph_norm = gnn.GraphNorm(embed_dim)
        self.graph_selfattention = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.LeakyReLU(),
                                                                                    nn.Dropout(p_dropout_attn),
                                                                                    nn.Linear(embed_dim, embed_dim*2),
                                                                                    nn.LeakyReLU(),
                                                                                    nn.Linear(embed_dim*2, 1),
                                                                                    AttnDropout(dropout_nodes),
                                                                                    ),
                                               GatedResidualSequential([nn.Linear(embed_dim, embed_dim*2),
                                                             nn.LeakyReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(embed_dim*2, embed_dim)], use_layernorm=False)) for i in range(n_pooling_heads)])
        self.graph_crossattention = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.LeakyReLU(),
                                                                                    nn.Dropout(p_dropout_attn),
                                                                                    nn.Linear(embed_dim, embed_dim*2),
                                                                                    nn.LeakyReLU(),
                                                                                    nn.Linear(embed_dim*2, 1),
                                                                                    AttnDropout(dropout_nodes),
                                                                                    ),
                                               GatedResidualSequential([
                                                             nn.Linear(embed_dim, embed_dim*2),
                                                             nn.LeakyReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(embed_dim*2, embed_dim)], use_layernorm=False)) for i in range(n_pooling_heads)])
        self.fc_s = nn.Sequential(GatedResidualSequential([
                                                           nn.Linear(embed_dim,
                                            embed_dim*2), 
                                nn.LeakyReLU(),
                                nn.Linear(embed_dim*2,
                                          embed_dim)], use_layernorm=True),
                                nn.Dropout(p=p_dropout_fc))
        self.fc_b = nn.Sequential(GatedResidualSequential([
                                                           nn.Linear(embed_dim,
                                            embed_dim*2), 
                                nn.LeakyReLU(),
                                nn.Linear(embed_dim*2,
                                          embed_dim)], use_layernorm=True),
                                nn.Dropout(p=p_dropout_fc))
        self.fc = nn.Sequential(nn.Sigmoid(),
                                GatedResidualSequential([
                                        nn.Linear(embed_dim, fc_hidden),
                                        nn.ReLU(),
                                        nn.Linear(fc_hidden, embed_dim),
                                        activation(),], use_layernorm=False),
                                nn.Linear(embed_dim, 1))
        self.target_dim = target_dim
    def forward(self,data,  *args, **kwargs):
        data = data.clone()
        genexp_ = self.geneencoder(data["expression"]).squeeze()
        node_embeddings = self.gat_encoder(data["x"], data["edge_index"], data["edge_attr"])
        graph_features = genexp_.new_zeros(genexp_.shape)
        for i in range(self.n_pooling_heads):
            graph_features += (self.graph_crossattention[i](
                node_embeddings + torch.repeat_interleave(F.relu(genexp_),
                                                          torch.bincount(data["batch"]), 0).squeeze(),
                data["batch"]) + self.graph_crossattention[i](node_embeddings, data["batch"]) + F.relu(genexp_))
        #graph_features = self.resnet(graph_features)
        graph_features_s = self.fc_s(graph_features)
        graph_features_b = self.fc_b(graph_features)
        sizes = graph_features_s.shape
        graph_features = graph_features_s.unsqueeze(2).expand(sizes[0], sizes[1], data["concs"].shape[1])
        graph_features = self.fc( graph_features_b.unsqueeze(1)  -\
                                 graph_features.transpose(2, 1) *\
                                 torch.clip(self.noise(data["concs"]), min = 0.0).unsqueeze(-1))
        return graph_features
    
class ARCANet(nn.Module):
    def __init__(self,
                 p_dropout_attn = 0.3,
                 p_dropout_fc = 0.4,
                 dropout_genes = 0.3,
                 dropout_nodes = 0.2,
                 embed_dim=512,
                 n_pooling_heads = 4,
                 gat_heads = 2,
                 gat_layers = 4,
                 target_dim = 166,
                 fc_hidden = 2048,
                 activation = 0,
                 gaussian_noise = 0,
                 **kwargs):
        super().__init__()
        if activation == 0:
            activation = nn.Sigmoid
        elif activation == 1:
            activation = nn.ELU
        elif activation == 2:
            activation = nn.ReLU
        else:
            activation = nn.Tanh
        self.noise = MultiplicativeGaussianNoise(gaussian_noise)
        self.geneencoder = GeneEncoder(2089, embed_dim, p_dropout=dropout_genes)
        self.n_pooling_heads = n_pooling_heads
        self.p_dropout_fc = p_dropout_fc
        self.gat_encoder = GATRes(init_dim = 79,
                                  hidden_dim = embed_dim,
                                  n_heads=gat_heads,
                                  concat=False,
                                  edge_dim=10,
                                  n_layers = gat_layers)
        self.graph_norm = gnn.GraphNorm(embed_dim)
        self.graph_crossattention = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.LeakyReLU(),
                                                                                    nn.Dropout(p_dropout_attn),
                                                                                    nn.Linear(embed_dim, embed_dim*2),
                                                                                    nn.LeakyReLU(),
                                                                                    nn.Linear(embed_dim*2, 1),
                                                                                    AttnDropout(dropout_nodes),
                                                                                    ),
                                               GatedResidualSequential([
                                                             nn.Linear(embed_dim, embed_dim*2),
                                                             nn.LeakyReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(embed_dim*2, embed_dim)], use_layernorm=False)) for i in range(n_pooling_heads)])
        self.fc_s = nn.Sequential(GatedResidualSequential([
                                                           nn.Linear(embed_dim,
                                            embed_dim*2), 
                                nn.LeakyReLU(),
                                nn.Linear(embed_dim*2,
                                          embed_dim)], use_layernorm=True),
                                nn.Dropout(p=p_dropout_fc))
        self.fc_b = nn.Sequential(GatedResidualSequential([
                                                           nn.Linear(embed_dim,
                                            embed_dim*2), 
                                nn.LeakyReLU(),
                                nn.Linear(embed_dim*2,
                                          embed_dim)], use_layernorm=True),
                                nn.Dropout(p=p_dropout_fc))
        self.fc = nn.Sequential(nn.Sigmoid(),
                                GatedResidualSequential([
                                        nn.Linear(embed_dim, fc_hidden),
                                        nn.ReLU(),
                                        nn.Linear(fc_hidden, embed_dim),
                                        activation(),], use_layernorm=False),
                                nn.Linear(embed_dim, 1))
        self.target_dim = target_dim
    def forward(self,data,  *args, **kwargs):
        data = data.clone()
        genexp_ = self.geneencoder(data["expression"]).squeeze()
        node_embeddings = self.gat_encoder(data["x"], data["edge_index"], data["edge_attr"])
        graph_features = genexp_.new_zeros(genexp_.shape)
        for i in range(self.n_pooling_heads):
            graph_features += (self.graph_crossattention[i](
                node_embeddings + torch.repeat_interleave(F.relu(genexp_),
                                                          torch.bincount(data["batch"]), 0).squeeze(),
                data["batch"]))
        graph_features += F.relu(genexp_)
        #graph_features = self.resnet(graph_features)
        graph_features_s = self.fc_s(graph_features)
        graph_features_b = self.fc_b(graph_features)
        sizes = graph_features_s.shape
        graph_features = graph_features_s.unsqueeze(2).expand(sizes[0], sizes[1], data["concs"].shape[1])
        graph_features = self.fc( graph_features_b.unsqueeze(1)  -\
                                 graph_features.transpose(2, 1) *\
                                 torch.clip(self.noise(data["concs"]), min = 0.0).unsqueeze(-1))
        return graph_features

class ARCANetNoGates(nn.Module):
    def __init__(self,
                 p_dropout_attn = 0.3,
                 p_dropout_fc = 0.4,
                 dropout_genes = 0.3,
                 dropout_nodes = 0.2,
                 embed_dim=512,
                 n_pooling_heads = 4,
                 gat_heads = 2,
                 gat_layers = 4,
                 target_dim = 166,
                 fc_hidden = 2048,
                 activation = 0,
                 gaussian_noise = 0,
                 **kwargs):
        super().__init__()
        if activation == 0:
            activation = nn.Sigmoid
        elif activation == 1:
            activation = nn.ELU
        elif activation == 2:
            activation = nn.ReLU
        else:
            activation = nn.Tanh
        self.noise = MultiplicativeGaussianNoise(gaussian_noise)
        self.geneencoder = GeneEncoder(2089, embed_dim, p_dropout=dropout_genes)
        self.n_pooling_heads = n_pooling_heads
        self.p_dropout_fc = p_dropout_fc
        self.gat_encoder = GATResOld(init_dim = 79,
                                  hidden_dim = embed_dim,
                                  n_heads=gat_heads,
                                  concat=False,
                                  edge_dim=10,
                                  n_layers = gat_layers)
        self.graph_norm = gnn.GraphNorm(embed_dim)
        self.graph_selfattention = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.LeakyReLU(),
                                                                                    nn.Dropout(p_dropout_attn),
                                                                                    nn.Linear(embed_dim, embed_dim*2),
                                                                                    nn.LeakyReLU(),
                                                                                    nn.Linear(embed_dim*2, 1),
                                                                                    AttnDropout(dropout_nodes),
                                                                                    ),
                                               GatedResidualSequentialOld([nn.Linear(embed_dim, embed_dim*2),
                                                             nn.LeakyReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(embed_dim*2, embed_dim)], use_layernorm=False)) for i in range(n_pooling_heads)])
        self.graph_crossattention = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.LeakyReLU(),
                                                                                    nn.Dropout(p_dropout_attn),
                                                                                    nn.Linear(embed_dim, embed_dim*2),
                                                                                    nn.LeakyReLU(),
                                                                                    nn.Linear(embed_dim*2, 1),
                                                                                    AttnDropout(dropout_nodes),
                                                                                    ),
                                               GatedResidualSequentialOld([
                                                             nn.Linear(embed_dim, embed_dim*2),
                                                             nn.LeakyReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(embed_dim*2, embed_dim)], use_layernorm=False)) for i in range(n_pooling_heads)])
        self.fc_s = nn.Sequential(GatedResidualSequentialOld([
                                                           nn.Linear(embed_dim,
                                            embed_dim*2), 
                                nn.LeakyReLU(),
                                nn.Linear(embed_dim*2,
                                          embed_dim)], use_layernorm=True),
                                nn.Dropout(p=p_dropout_fc))
        self.fc_b = nn.Sequential(GatedResidualSequentialOld([
                                                           nn.Linear(embed_dim,
                                            embed_dim*2), 
                                nn.LeakyReLU(),
                                nn.Linear(embed_dim*2,
                                          embed_dim)], use_layernorm=True),
                                nn.Dropout(p=p_dropout_fc))
        self.fc = nn.Sequential(nn.Sigmoid(),
                                GatedResidualSequentialOld([
                                        nn.Linear(embed_dim, fc_hidden),
                                        nn.ReLU(),
                                        nn.Linear(fc_hidden, embed_dim),
                                        activation(),], use_layernorm=False),
                                nn.Linear(embed_dim, 1))
        self.target_dim = target_dim
    def forward(self,data,  *args, **kwargs):
        data = data.clone()
        genexp_ = self.geneencoder(data["expression"]).squeeze()
        node_embeddings = self.gat_encoder(data["x"], data["edge_index"], data["edge_attr"])
        graph_features = genexp_.new_zeros(genexp_.shape)
        for i in range(self.n_pooling_heads):
            graph_features += (self.graph_crossattention[i](
                node_embeddings + torch.repeat_interleave(F.relu(genexp_),
                                                          torch.bincount(data["batch"]), 0).squeeze(),
                data["batch"]) + self.graph_crossattention[i](node_embeddings, data["batch"]) + F.relu(genexp_))
        #graph_features = self.resnet(graph_features)
        graph_features_s = self.fc_s(graph_features)
        graph_features_b = self.fc_b(graph_features)
        sizes = graph_features_s.shape
        graph_features = graph_features_s.unsqueeze(2).expand(sizes[0], sizes[1], data["concs"].shape[1])
        graph_features = self.fc( graph_features_b.unsqueeze(1)  -\
                                 graph_features.transpose(2, 1) *\
                                 torch.clip(self.noise(data["concs"]), min = 0.0).unsqueeze(-1))
        return graph_features
    
class SigmoidModel(nn.Module):
    def __init__(self,
                 n_drugs,
                 n_lines,
                 device,
                 parameters = 4):
        super().__init__()
        if parameters < 4:
            self.B1l = torch.zeros([n_lines]).to(device)
        else:
            self.B1l = nn.Parameter(torch.randn([n_lines]) * 1)
        if parameters < 3:
            self.B2l = torch.ones([n_lines]).to(device)
        else:
            self.B2l = nn.Parameter(torch.randn([n_lines]) * 1)
        self.B3l = nn.Parameter(torch.randn([n_lines]) * 1)
        self.B4l = nn.Parameter(torch.randn([n_lines]) * 1)
        self.B4ld = nn.Parameter(torch.randn([n_lines, n_drugs]) * 1)
    def forward(self, d, l, c):
        B1 = self.B1l[l].unsqueeze(1)
        B2 = self.B2l[l].unsqueeze(1)
        B3 = self.B3l[l].unsqueeze(1)
        B4 = self.B4l[l].unsqueeze(1) + self.B4ld[l, d].unsqueeze(1)
        return B1 + (B2 - B1)*torch.sigmoid(B4*(c - B3))