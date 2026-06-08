import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch_geometric
from torch_scatter import scatter, scatter_softmax
from torch_geometric.nn import radius_graph, TransformerConv, global_add_pool, SAGPooling
import torch.nn.utils.prune as prune


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_hidden)
        self.W11 = nn.Linear(3 * num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.act(self.W11(h_EV))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E
    
    
class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()
        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )
    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class GNNLayer(nn.Module):
    def __init__(self, num_hidden, context=False, dropout=0.2, num_heads=4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_hidden)
        self.attention = TransformerConv(in_channels=num_hidden, 
                                         out_channels=int(num_hidden / num_heads),
                                         heads=num_heads, 
                                         edge_dim=num_hidden, 
                                         dropout=dropout)
        self.edge_update = EdgeMLP(num_hidden, dropout)
        if context:
           self.context = Context(num_hidden)
        self.saving_attn = None

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh, (edge_index_att, attn_weights) = self.attention(h_V, edge_index, h_E,
                                                            return_attention_weights=True)
        self.saving_attn = (edge_index_att, attn_weights)
        h_V = self.norm(h_V + self.dropout(dh))
        h_E = self.edge_update(h_V, edge_index, h_E)
        
        if hasattr(self, 'context'):
            h_V = self.context(h_V, batch_id)
            
        return h_V, h_E
    
    
class GlobalAttentionPool(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.gate_nn = torch.nn.Linear(in_channels, 1)
        self.temp = torch.nn.Parameter(torch.ones(1))

        self.saving_attn = None
    
    def forward(self, x, batch, binding_mask):
        # x: [num_nodes, in_channels]
        x = self.bn(x)
        score = self.gate_nn(x)

        attention_weights = scatter_softmax(score + torch.log(binding_mask.unsqueeze(1)), batch, dim=0) 
        self.saving_attn = attention_weights.detach()
        
        graph_embedding = global_add_pool(x * attention_weights, batch)
        return graph_embedding
    
    
class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers=4, drop_rate=0.2, context=False):
        super(Graph_encoder, self).__init__()
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.LayerNorm(hidden_dim)
        self.norm_edges = nn.LayerNorm(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4, context=context)
            for _ in range(num_layers))

    def forward(self, h_V, edge_index, h_E, batch_id):
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V
    
    
class GraphModel(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim,
                 hidden_dim, num_layers, dropout,
                 device, pool=None, out_dim=2,
                 node_geo_features=None, edge_geo_features=None):
        super().__init__()
        self.h_V_dropout = nn.Dropout(dropout / 2)
        if out_dim == 2:
            self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, 
                                               hidden_dim=hidden_dim, num_layers=num_layers, drop_rate=dropout,
                                               context=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
                nn.ELU(),
                nn.Linear(hidden_dim // 2, 2, bias=True)
        )
        else:
            self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, 
                                               hidden_dim=hidden_dim, num_layers=num_layers, drop_rate=dropout,
                                               context=False)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ELU(),
                nn.Linear(hidden_dim // 2, out_dim),
                nn.Sigmoid()
            )

        self.pool = pool
        if pool is not None:
            if pool == 'attn':
                self.pooling_layer = GlobalAttentionPool(hidden_dim)
            elif pool == 'sag':
                self.pooling_layer = SAGPooling(in_channels=hidden_dim, ratio=0.5, multiplier=1.0)

        self.node_geo_features = node_geo_features
        self.edge_geo_features = edge_geo_features
        
        self.to(device)

    def forward(self, X, h_V, edge_index, batch_id, binding_mask):
        h_V_geo, h_E = get_geo_feat(X, edge_index, self.node_geo_features, self.edge_geo_features)
        
        h_V = self.h_V_dropout(h_V)
        if h_V_geo is not None:
            h_V = torch.cat([h_V, h_V_geo], dim=-1)

        h_V = self.Graph_encoder(h_V, edge_index, h_E, batch_id)
        if self.pool is not None:
            if self.pool == 'sum':
                h_V = scatter_add(h_V, batch_id, dim=0)
            elif self.pool == 'mean':
                h_V = scatter_mean(h_V, batch_id, dim=0)
            elif self.pool == 'attn':
                h_V = self.pooling_layer(h_V, batch_id, binding_mask)
            elif self.pool == 'sag':
                x, _, _, batch_reduced, _, _ = self.pooling_layer(h_V, edge_index, batch=batch_id)
                h_V = global_add_pool(x, batch_reduced)
                
        logits = self.head(h_V)
        return logits
    
    
def build_model(args):
    model = GraphModel(
        node_input_dim=args.node_input_dim,
        edge_input_dim=args.edge_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pool=args.pool,
        out_dim=args.out_dim,
        node_geo_features=['node_angles', 'node_dist', 'node_direction'],
        edge_geo_features=['pos_embeddings', 'edge_orientation', 'edge_dist', 'edge_direction'],
        device=args.device
    )
    return model
    
    
def get_geo_feat(X, edge_index, node_geo_features=None, edge_geo_features=None):
    pos_embeddings = positional_embeddings(edge_index)
    node_angles = get_angle(X)
    node_dist, edge_dist = get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = get_direction_orientation(X, edge_index)
    nodes = {'node_angles': node_angles, 'node_dist': node_dist, 'node_direction': node_direction}
    edges = {'pos_embeddings': pos_embeddings, 'edge_orientation': edge_orientation, 'edge_dist': edge_dist, 'edge_direction': edge_direction}
    
    if node_geo_features is not None:
        geo_node_feat = torch.cat([nodes[feature_name] for feature_name in node_geo_features], dim=-1)
    else:
        geo_node_feat = None
    if edge_geo_features is not None:
        geo_edge_feat = torch.cat([edges[feature_name] for feature_name in edge_geo_features], dim=-1)
    else:
        geo_edge_feat = None
    return geo_node_feat, geo_edge_feat


def positional_embeddings(edge_index, num_embeddings=16):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE


def compute_dihedral_atan2(a, b, c, d):
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = F.normalize(b1, dim=-1)
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    return torch.atan2(y, x)
    
    
def get_angle(X, eps=1e-7):
    num_res = X.shape[0]
    if num_res < 2:
        return torch.zeros(num_res, 12, device=X.device, dtype=X.dtype)

    atoms = X[:, :3].reshape(-1, 3)

    dX = atoms[1:] - atoms[:-1]
    U = F.normalize(dX, dim=-1)

    #! For RNA, P, C4', N1/N9
    N = X[:, 0]
    CA = X[:, 1]
    C = X[:, 2]

    dihedrals = torch.zeros(num_res, 3, device=X.device, dtype=X.dtype)
    # phi[:-1]: C_i, N_i+1, CA_i+1, C_i+1
    dihedrals[1:, 0] = compute_dihedral_atan2(C[:-1], N[1:], CA[1:], C[1:])
    # psi[:-1]
    dihedrals[:-1, 1] = compute_dihedral_atan2(N[:-1], CA[:-1], C[:-1], N[1:])
    # omega[1:]
    dihedrals[:-1, 2] = compute_dihedral_atan2(CA[:-1], C[:-1], N[1:], CA[1:])
    dihedral_feats = torch.cat([torch.cos(dihedrals), torch.sin(dihedrals)], dim=1)

    # Bond/chain angles (alpha, beta, gamma)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    cosD = (u_2 * u_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angle_feats = torch.cat([torch.cos(D), torch.sin(D)], dim=1)
    return torch.cat([dihedral_feats, bond_angle_feats], dim=1)


def rbf(D, D_min=0., D_max=20., D_count=16):
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def get_distance(X, edge_index):
    """
    Compute distance inner a residue no matter prot residue or RNA residue
    Args:
        X (torch.tensor): (N + M, 5, 3)
            For prot residue: 
                [N, Ca, C, O, R]
            For RNA residue:
                [C4', P, N1/N9, O5', C5']
        rna_mask (N + M, )
        edge_index (K, 2)
    """
    #! calcualte node distance
    dist_vec = X[:, :, None, :] - X[:, None, :, :]  # (L, 5, 5, 3)
    dist_vec = dist_vec.norm(dim=-1) # (L, 5, 5)
    diag_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    dist_vec = dist_vec[:, diag_mask]
    node_dist = rbf(dist_vec).reshape(X.shape[0], -1)
    
    #! calcualte edge distance
    edge_vec = X[edge_index[0], :, None, :] - X[edge_index[1], None, :, :]
    edge_vec = edge_vec.norm(dim=-1) # (K, 5, 5)
    edge_dist = rbf(edge_vec).reshape(edge_vec.shape[0], -1)  # (K, 25 * 16)
    return node_dist, edge_dist


def get_direction_orientation(X, edge_index): # N, CA, C, O, R
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3]

    node_j, node_i = edge_index

    t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1) # [L, 4 * 3]

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3] 
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    edge_orientation = quaternions(r) # [E, 4]

    return node_direction, edge_direction, edge_orientation


def quaternions(R):
    """ 
    R [E,3,3]
    Q [E,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes

    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q