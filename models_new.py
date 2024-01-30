import torch
from torch import nn
from torch.nn import functional
from abc import ABC, abstractmethod
import torch_geometric
from torch_geometric import nn as gnn
import numpy as np
from torch.nn import functional as F

class FusionModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def forward(x, z, self):
        """
        x: expected dimension [BATCH, F_x]
        z: expected dimension [BATCH, n_z]
        output: expected dimension [BATCH, n_z, F_x]
        """
        pass
class LatentHillFusionModule(FusionModule):
    def __init__(self, init_dim, hidden_dim, dropout=0.0, use_norm = True):
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            norm1 = nn.BatchNorm1d(init_dim, eps=1e-05)
            norm2 = nn.BatchNorm1d(init_dim, eps=1e-05)
        else:
            norm1 = nn.Identity()
            norm2 = nn.Identity()
        self.norm1 = norm1
        self.norm2 = norm2
        self.mlp_bias = nn.Sequential(nn.Linear(init_dim, hidden_dim),
                                      nn.Dropout1d(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, init_dim),
                                      norm1)
        self.mlp_slope = nn.Sequential(nn.Linear(init_dim, hidden_dim),
                                      nn.Dropout1d(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, init_dim),
                                      norm2)
    def forward(self, x, z):
        h_bias = (self.mlp_bias(x) + x)/2
        h_slope = (self.mlp_slope(x) + x)/2
        # if False:
        #     sizes = h_slope.shape
        #     graph_features = h_slope.unsqueeze(2).expand(sizes[0], sizes[1], z.shape[1])
        #     return torch.sigmoid( h_bias.unsqueeze(1)  -\
        #                              graph_features.transpose(2, 1) *\
        #                              torch.clip(z, min = 0.0).unsqueeze(-1))
        return torch.sigmoid(h_bias.unsqueeze(1)-(z.unsqueeze(2)*h_slope.unsqueeze(1)))
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()
        
    
class ConcatFusionModule(FusionModule):
    def __init__(self, init_dim, hidden_dim, dropout=0.0, use_norm = True):
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            norm1 = nn.BatchNorm1d(init_dim, eps=1e-05)
        else:
            norm1 = nn.Identity()
        self.norm1 = norm1
        self.mlp_fusion = nn.Sequential(nn.Linear(init_dim+1, hidden_dim),
                                      nn.Dropout1d(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, init_dim),
                                      norm1,)
    def forward(self, x, z):
        return self.mlp_fusion(torch.cat([x.unsqueeze(1).repeat(1, z.size(1), 1), z.unsqueeze(2)], -1))
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
    
class DeepSplineFusionModule(FusionModule):
    def __init__(self, init_dim, hidden_dim, dropout=0.0, n_knots=4, use_norm = True):
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            norm1 = nn.BatchNorm1d(hidden_dim, eps=1e-05)
            norm2 = nn.BatchNorm1d(init_dim * n_knots, eps=1e-05)
        else:
            norm1 = nn.Identity()
            norm2 = nn.Identity()
        self.norm1 = norm1
        self.norm2 = norm2
        self.init_dim = init_dim
        self.n_knots = n_knots
        self.mlp_z = nn.Sequential(nn.Linear(1, hidden_dim),
                                      nn.Dropout1d(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, n_knots),
                                      norm1)
        self.mlp_x = nn.Sequential(nn.Linear(init_dim, hidden_dim),
                                      nn.Dropout1d(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, init_dim * n_knots),
                                      norm2)
    def forward(self, x, z):
        zs = self.mlp_z(z.unsqueeze(-1)).unsqueeze(1)
        xs = self.mlp_x(x).reshape(x.size(0), -1, self.n_knots).unsqueeze(2)
        out = zs.mul(xs)
        return out.sum(-1).transpose(1, 2)
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm1.reset_running_stats()
            self.norm2.reset_running_stats()

class CrossAttnPooling(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_heads,
                 p_dropout,
                 res = True,
                 add_q=True,
                 **kwargs):
        super().__init__()
        self.g_c = gnn.global_max_pool
        self.pool = nn.MultiheadAttention(output_dim, num_heads, p_dropout, batch_first=True)
        self.t_q = nn.Sequential(nn.Linear(input_dim,output_dim,), nn.ReLU(), nn.Linear(output_dim,output_dim,))
        self.add_q = add_q
        self.tdb = tdb = torch_geometric.utils.to_dense_batch
        self.res = res
    def forward(self, 
                x,
                y,
                batch_drugs = None,
                batch_lines = None,
                *args, **kwargs):
        if batch_drugs is None:
            batch_drugs = x.new_zeros(x.size(0)).long()
        if batch_lines is None:
            batch_lines = x.new_zeros(y.size(0)).long()
        out1 = self.tdb(x, batch_drugs)
        keys = values= out1[0]
        mask_drugs = out1[1]
        y = self.t_q(y)
        out2 = self.tdb(y, batch_lines)
        query = out2[0]
        mask_lines = out2[1]
        x_, _ = self.pool(query, keys, values, key_padding_mask = ~mask_drugs)
        batch = out = query = keys = values = ""
        if self.add_q:
            return x_[mask_lines].squeeze() + y
        else:
            return x_[mask_lines].squeeze()
    
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

class GatedAttnPooling(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 p_dropout_attn,
                 dropout_nodes,
                 n_pooling_heads,
                 res = True,
                 add_q=True,
                 **kwargs):
        super().__init__()
        self.n_pooling_heads = n_pooling_heads
        self.pool = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.ReLU(),
                                                                                    nn.Dropout(p_dropout_attn),
                                                                                    nn.Linear(input_dim, hidden_dim,),
                                                                                    nn.ReLU(),
                                                                                    nn.Linear(hidden_dim, 1),
                                                                                    AttnDropout(dropout_nodes),
                                                                                    ),
                                               nn.Sequential(nn.Linear(input_dim, hidden_dim,),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, output_dim))) for i in range(n_pooling_heads)])
        self.tdb = tdb = torch_geometric.utils.to_dense_batch
    def forward(self, 
                x,
                y,
                batch_drugs = None,
                batch_lines = None,
                *args, **kwargs):
        if batch_drugs is None:
            batch_drugs = x.new_zeros(x.size(0)).long()
        if batch_lines is None:
            batch_lines = x.new_zeros(y.size(0)).long()
        out1 = self.tdb(x, batch_drugs)
        drugs_batched = out1[0]
        mask_drugs = out1[1]
        out2 = self.tdb(y, batch_lines)
        lines_batched = out2[0]
        mask_lines = out2[1]
        x_comb = drugs_batched + lines_batched
        x_comb = x_comb[mask_drugs]
        for i in range(self.n_pooling_heads):
            if i == 0:
                x = self.pool[i](x_comb, batch_drugs)
            else:
                x += self.pool[i](x_comb, batch_drugs)
        return x    

class MLPEncoder(nn.Module):
    def __init__(self, init_dim,
                 bottleneck_dim,
                 p_dropout=0.0,
                 p_dropout_hidden=0.0):
        super().__init__()
        self.layers = nn.Sequential(nn.Dropout(p=p_dropout),
                            nn.Linear(init_dim, bottleneck_dim),
                            nn.Dropout(p=p_dropout_hidden))
        self.layers.apply(init_weights)
    def forward(self, x):
        return self.layers(x)

class MultiModelFP(nn.Module):
    def __init__(self,
                 cell_encoder,
                 drug_encoder,
                 drug_cell_fusion,
                 fusion_module,
                 embed_dim=512,
                 fc_hidden = 2048,
                 gaussian_noise = 0,
                 dropout_fc=0.0,
                 use_normalization = True,
                 use_normalization2 = True,
                 linear_head = False,
                 transform_log_conc = False,
                 activation_fn = "tanh",
                 **kwargs):
        super().__init__()
        if activation_fn == "tanh":
            activation_fn = nn.Tanh
        elif activation_fn == "relu":
            activation_fn = nn.ReLU
        elif activation_fn == "sigmoid":
            activation_fn = nn.Sigmoid
        self.cell_encoder = cell_encoder
        self.drug_encoder = drug_encoder
        self.dc_fusion = drug_cell_fusion
        self.fusion_module = fusion_module
        self.norm = nn.BatchNorm1d(embed_dim, eps=1e-05)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_norm = use_normalization
        self.use_norm2 = use_normalization2
        self.log_conc = transform_log_conc
        if linear_head:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Sequential(nn.Sequential(
                                            nn.Linear(embed_dim, fc_hidden),
                                            nn.ReLU(),
                                            nn.Dropout1d(dropout_fc),
                                            nn.Linear(fc_hidden, embed_dim),
                                            activation_fn()))
        self.lin = nn.Linear(embed_dim, 1)
    def forward(self,data,  *args, **kwargs):
        data = data.clone()
        genexp = self.cell_encoder(data["expression"]).squeeze()
        drug_x = self.drug_encoder(data["fingerprint"])
        x = self.dc_fusion(genexp, drug_x)
        if self.use_norm:
            x = self.norm(x)
        if self.log_conc:
            zs = torch.log(data["z"])
        else:
            zs = data["z"]
        x = self.fusion_module(x, zs)
        x = (self.fc(x) + x)/2
        if self.use_norm2:
            x = self.norm2(x)
        return self.lin(x)
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm.reset_running_stats()
        if self.fusion_module.use_norm:
            self.fusion_module.reset_batchnorm()    

class MultiModel(nn.Module):
    def __init__(self,
                 cell_encoder,
                 drug_encoder,
                 graph_crossattention,
                 fusion_module,
                 embed_dim=512,
                 fc_hidden = 2048,
                 gaussian_noise = 0,
                 dropout_fc=0.0,
                 use_normalization = True,
                 use_normalization2 = True,
                 linear_head = False,
                 transform_log_conc = False,
                 activation_fn = "tanh",
                 **kwargs):
        super().__init__()
        if activation_fn == "tanh":
            activation_fn = nn.Tanh
        elif activation_fn == "relu":
            activation_fn = nn.ReLU
        elif activation_fn == "sigmoid":
            activation_fn = nn.Sigmoid
        self.cell_encoder = cell_encoder
        self.drug_encoder = drug_encoder
        self.graph_crossattention = graph_crossattention
        self.fusion_module = fusion_module
        self.norm = nn.BatchNorm1d(embed_dim, eps=1e-05)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_norm = use_normalization
        self.use_norm2 = use_normalization2
        self.log_conc = transform_log_conc
        if linear_head:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Sequential(nn.Sequential(
                                            nn.Linear(embed_dim, fc_hidden),
                                            nn.ReLU(),
                                            nn.Dropout1d(dropout_fc),
                                            nn.Linear(fc_hidden, embed_dim),
                                            activation_fn()))
        self.lin = nn.Linear(embed_dim, 1)
    def forward(self,data,  *args, **kwargs):
        data = data.clone()
        genexp = self.cell_encoder(data["expression"]).squeeze()
        node_embeddings = self.drug_encoder(data["x"], data["edge_index"], data["edge_attr"], data["batch"])
        batch_lines = data["edge_index"].new_tensor(np.arange(len(data["y"])))
        graph_features = (self.graph_crossattention(node_embeddings, genexp, data["batch"], batch_lines) + genexp)/2
        if self.use_norm:
            graph_features = self.norm(graph_features)
        if self.log_conc:
            zs = torch.log(data["z"])
        else:
            zs = data["z"]
        graph_features = self.fusion_module(graph_features, zs)
        graph_features = (self.fc(graph_features) + graph_features)/2
        if self.use_norm2:
            graph_features = self.norm2(graph_features)
        return self.lin(graph_features)
    def reset_batchnorm(self):
        if self.use_norm:
            self.norm.reset_running_stats()
        if self.fusion_module.use_norm:
            self.fusion_module.reset_batchnorm()
        


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class GNNTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_heads,
                 p_dropout,
                 res = True,
                 **kwargs):
        super().__init__()
        self.g_c = gnn.global_max_pool
        self.pool = nn.MultiheadAttention(output_dim, num_heads, p_dropout, batch_first=True)
        self.tdb = tdb = torch_geometric.utils.to_dense_batch
        self.res = res
    def forward(self,x, batch = None,*args, **kwargs):
        if batch is None:
            batch = x.new_zeros(x.size(0)).long()
        out = self.tdb(x, batch)
        query = keys = values= out[0]
        mask = out[1]
        x_, _ = self.pool(query, keys, values, key_padding_mask = ~mask)
        batch = out = query = keys = values = ""
        if self.res:
            return (x.squeeze()+(x_[mask].squeeze()))/2
        else:
            return x_[mask].squeeze()

class GATRes(nn.Module):
    def __init__(
        
        self,
        init_dim = 9,
        hidden_dim = 128,
        edge_dim = 3,
        n_layers = 4,
        n_transformers = 1,
        n_heads = 4,
        dropout = 0,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.init_gat = gnn.GATv2Conv(init_dim, hidden_dim, edge_dim=edge_dim, heads=1)
        self.transformers = GNNTransformer(hidden_dim, hidden_dim, n_heads, dropout, n_transformers)
        self.gats = nn.ModuleList([gnn.GATv2Conv(hidden_dim, hidden_dim, concat=False, edge_dim=edge_dim, heads=n_heads) for i in range(n_layers - 1)])
    def forward(self, x, edge_index,edge_weight = None, batch=None):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x = self.init_gat(x, edge_index, edge_weight)
            for i in range(self.n_layers - 1):
                x = (self.gats[i](F.relu(x), edge_index, edge_weight) + x)/2
            return self.transformers(x, batch)