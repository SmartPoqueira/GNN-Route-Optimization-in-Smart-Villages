# model.py

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree, softmax

def get_activation_fn(name: str):
    """
    Returns the activation module corresponding to the given name.
    """
    name_lower = name.lower()
    if name_lower == 'relu':
        return nn.ReLU()
    elif name_lower == 'leakyrelu':
        return nn.LeakyReLU()
    elif name_lower == 'gelu':
        return nn.GELU()
    else:
        return nn.ReLU()

def build_mlp_block(input_dim: int, hidden_dim: int, n_layers: int, activation: str, dropout_rate: float = 0.5, use_batchnorm: bool = True, is_final: bool = False):
    """
    Builds a sequential MLP block.
    """
    layers = []
    curr_dim = input_dim
    for i in range(n_layers):
        if is_final and i == n_layers - 1:
            layers.append(nn.Linear(curr_dim, hidden_dim))
        else:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(get_activation_fn(activation))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            curr_dim = hidden_dim
    return nn.Sequential(*layers)

# ----------------- CUSTOM GNN LAYERS -----------------

class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
        
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class CustomEGCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int):
        super().__init__(aggr='add')
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_j, edge_attr):
        edge_mod = torch.sigmoid(self.lin_edge(edge_attr))
        return x_j * edge_mod

class CustomSAGEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggregation: str = 'mean'):
        super().__init__(aggr=aggregation)
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return self.lin_l(x) + self.lin_r(out)
        
    def message(self, x_j):
        return x_j

class CustomESAGEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int, aggregation: str = 'mean'):
        super().__init__(aggr=aggregation)
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_channels, in_channels)
        
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin_l(x) + self.lin_r(out)
        
    def message(self, x_j, edge_attr):
        edge_feat = self.lin_edge(edge_attr)
        return x_j + edge_feat

class CustomGGNNConv(MessagePassing):
    def __init__(self, channels: int):
        super().__init__(aggr='add')
        self.lin = nn.Linear(channels, channels)
        self.gru = nn.GRUCell(channels, channels)
        
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return self.gru(out, x)
        
    def message(self, x_j):
        return self.lin(x_j)

class CustomEGGNNConv(MessagePassing):
    def __init__(self, channels: int, edge_channels: int):
        super().__init__(aggr='add')
        self.lin = nn.Linear(channels, channels)
        self.lin_edge = nn.Linear(edge_channels, channels)
        self.gru = nn.GRUCell(channels, channels)
        
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.gru(out, x)
        
    def message(self, x_j, edge_attr):
        edge_gating = torch.sigmoid(self.lin_edge(edge_attr))
        return self.lin(x_j) * edge_gating

class CustomEdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
        
    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j - x_i], dim=-1))

class CustomMPNNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int, msg_mlp_layers: int, activation_name: str):
        super().__init__(aggr='mean')
        self.msg_mlp = build_mlp_block(
            input_dim=in_channels + edge_channels,
            hidden_dim=out_channels,
            n_layers=msg_mlp_layers,
            activation=activation_name
        )
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_j, edge_attr):
        combined = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_mlp(combined)

class CustomGATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add', node_dim=0)
        self.lin = nn.Linear(in_channels, out_channels)
        self.attn_linear = nn.Linear(2 * out_channels, 1)
        
    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)
        
    def message(self, x_i, x_j, index, ptr, size_i):
        attn_input = torch.cat([x_i, x_j], dim=-1)
        attn_score = self.attn_linear(attn_input)
        attn_score = torch.tanh(attn_score)
        alpha = softmax(attn_score, index, ptr, num_nodes=size_i)
        return alpha * x_j

class CustomEGATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int, msg_mlp_layers: int, activation_name: str, aggregation: str = 'mean', use_residual: bool = True):
        super().__init__(aggr=aggregation, node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.use_residual = use_residual
        
        self.activation_fn = get_activation_fn(activation_name)
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_channels, out_channels)
        
        self.attn_linear = nn.Linear(3 * out_channels, 1)
        
        if msg_mlp_layers > 0:
            self.msg_mlp = build_mlp_block(
                input_dim=2 * out_channels,
                hidden_dim=out_channels,
                n_layers=msg_mlp_layers,
                activation=activation_name
            )
        else:
            self.msg_mlp = nn.Linear(2 * out_channels, out_channels)
            
        self.edge_update_mlp = nn.Linear(3 * out_channels, out_channels)
        self.node_update_mlp = nn.Linear(2 * out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        h_node = self.lin_node(x)
        h_edge = self.lin_edge(edge_attr)
        
        out_nodes = self.propagate(edge_index, x=h_node, edge_attr=h_edge)
        
        row, col = edge_index
        edge_inputs = torch.cat([h_node[row], h_node[col], h_edge], dim=-1)
        new_edge_attr = self.edge_update_mlp(edge_inputs)
        new_edge_attr = self.activation_fn(new_edge_attr)
        if self.use_residual:
            new_edge_attr = new_edge_attr + h_edge
            
        new_x = self.node_update_mlp(torch.cat([h_node, out_nodes], dim=-1))
        new_x = self.activation_fn(new_x)
        if self.use_residual:
            new_x = new_x + h_node
            
        return new_x, new_edge_attr
        
    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        attn_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        attn_score = self.attn_linear(attn_input)
        attn_score = torch.tanh(attn_score)
        alpha = softmax(attn_score, index, ptr, num_nodes=size_i)
        
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        if isinstance(self.msg_mlp, nn.Sequential):
            msg = self.msg_mlp(msg_input)
        else:
            msg = self.msg_mlp(msg_input)
        return alpha * msg

# ----------------- UNIFIED GNN CLASSIFIER -----------------

class GNNClassifier(nn.Module):
    def __init__(self, model_name: str, num_node_features: int, num_edge_features: int, model_params: dict):
        super().__init__()
        self.model_name = model_name.upper()
        self.hidden_dim = model_params.get('hidden_dim', 128)
        dropout_rate = model_params.get('dropout', 0.5)
        use_batchnorm = model_params.get('use_batchnorm', True)
        activation = model_params.get('activation', 'ReLU')
        aggregation = model_params.get('aggregation', 'mean')
        
        n_node_proj = model_params.get('node_mlp_layers', 1)
        n_edge_proj = model_params.get('edge_mlp_layers', 1)
        
        self.node_proj = build_mlp_block(num_node_features, self.hidden_dim, n_node_proj, activation, dropout_rate, use_batchnorm)
        self.edge_proj = build_mlp_block(num_edge_features, self.hidden_dim, n_edge_proj, activation, dropout_rate, use_batchnorm)
        
        num_layers = model_params.get('gnn_layers', 1)
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if self.model_name == 'GCN':
                self.gnn_layers.append(CustomGCNConv(self.hidden_dim, self.hidden_dim))
            elif self.model_name == 'E-GCN':
                self.gnn_layers.append(CustomEGCNConv(self.hidden_dim, self.hidden_dim, self.hidden_dim))
            elif self.model_name == 'GRAPHSAGE':
                self.gnn_layers.append(CustomSAGEConv(self.hidden_dim, self.hidden_dim, aggregation))
            elif self.model_name == 'E-GRAPHSAGE':
                self.gnn_layers.append(CustomESAGEConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, aggregation))
            elif self.model_name == 'GGNN':
                self.gnn_layers.append(CustomGGNNConv(self.hidden_dim))
            elif self.model_name == 'E-GGNN':
                self.gnn_layers.append(CustomEGGNNConv(self.hidden_dim, self.hidden_dim))
            elif self.model_name == 'EDGECONV':
                self.gnn_layers.append(CustomEdgeConv(self.hidden_dim, self.hidden_dim))
            elif self.model_name == 'MPNN':
                n_msg = model_params.get('message_mlp_layers', 1)
                self.gnn_layers.append(CustomMPNNConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, n_msg, activation))
            elif self.model_name == 'GAT':
                self.gnn_layers.append(CustomGATConv(self.hidden_dim, self.hidden_dim))
            elif self.model_name == 'E-GAT':
                n_msg = model_params.get('message_mlp_layers', 1)
                use_residual = model_params.get('use_residual', True)
                self.gnn_layers.append(CustomEGATConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, n_msg, activation, aggregation, use_residual))
            else:
                raise ValueError(f"Unknown model name: {self.model_name}")
                
        n_final = model_params.get('final_mlp_layers', 2)
        self.final_mlp = build_mlp_block(self.hidden_dim, 1, n_final, activation, dropout_rate, use_batchnorm, is_final=True)
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_proj(x)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = self.edge_proj(edge_attr)
        else:
            edge_attr = torch.zeros((edge_index.size(1), self.hidden_dim), device=x.device)
            
        for layer in self.gnn_layers:
            if self.model_name in ['E-GCN', 'E-GRAPHSAGE', 'E-GGNN', 'MPNN']:
                x = layer(x, edge_index, edge_attr)
            elif self.model_name == 'E-GAT':
                x, edge_attr = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)
                
        x = global_mean_pool(x, batch)
        return self.final_mlp(x)

# For backwards compatibility with the NAS pipeline
class ImprovedMPNN_NAS(nn.Module):
    def __init__(self, num_node_features: int, num_edge_features: int, model_params: dict):
        super().__init__()
        # Determine the model name from params, default to E-GAT
        model_name = model_params.get('model_name', 'E-GAT')
        self.classifier = GNNClassifier(model_name, num_node_features, num_edge_features, model_params)
        
    def forward(self, x, edge_index, edge_attr, batch):
        return self.classifier(x, edge_index, edge_attr, batch)
