# improvedMPNN_nas.py

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear, ReLU, Dropout, BatchNorm1d, LayerNorm
from torch_geometric.nn import MessagePassing, global_mean_pool, GATConv
from torch_geometric.utils import add_self_loops, remove_self_loops

def build_mlp(
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout_rate: float = 0.5,
    use_batchnorm: bool = True,
    is_final_mlp: bool = False,
    use_residual: bool = False
):
    """
    Builds a sequential MLP with 'n_layers' blocks of the form:
      [Linear -> ReLU -> (BatchNorm1d) -> Dropout]

    - If is_final_mlp=True, the last generated layer is: Linear(hidden_dim, 1)
      without subsequent ReLU/BN/Dropout.
    - If use_residual=True, a residual connection is added between the input
      and the output of the MLP.

    Examples:
    - n_layers=2, is_final_mlp=False:
      1) Linear(input_dim, hidden_dim) -> ReLU -> BN -> Dropout
      2) Linear(hidden_dim, hidden_dim) -> ReLU -> BN -> Dropout

    - n_layers=2, is_final_mlp=True:
      1) Linear(input_dim, hidden_dim) -> ReLU -> BN -> Dropout
      2) Linear(hidden_dim, 1)
    """
    layers = []
    current_dim = input_dim

    for i in range(n_layers):
        if is_final_mlp and i == n_layers - 1:
            layers.append(Linear(current_dim, 1))
        else:
            layers.append(Linear(current_dim, hidden_dim))
            layers.append(ReLU())
            if use_batchnorm:
                layers.append(BatchNorm1d(hidden_dim))
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
            current_dim = hidden_dim

    mlp = Seq(*layers)

    if use_residual and input_dim == hidden_dim and n_layers > 0:
        # Add a residual connection if dimensions match
        return Residual(mlp)
    else:
        return mlp

class Residual(nn.Module):
    """
    Residual layer that adds the input to the output of a sublayer.
    """
    def __init__(self, sublayer):
        super(Residual, self).__init__()
        self.sublayer = sublayer

    def forward(self, x):
        return x + self.sublayer(x)

class AttentionLayer(nn.Module):
    """
    Attention layer that weights node features.
    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        # x: [num_nodes, hidden_dim]
        # edge_index: [2, num_edges]
        # batch: [num_nodes]
        # Compute attention scores for each node
        attn_scores = self.attention(x)  # [num_nodes, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)  # [num_nodes, 1]
        return x * attn_weights  # [num_nodes, hidden_dim]

class ImprovedMPNN_NAS(MessagePassing):
    """
    Flexible Improved MPNN that dynamically defines the structure
    (number of layers) in each internal MLP: node_mlp, edge_mlp, message_mlp, final_mlp.
    Supports variable hidden_dim, dropout, aggregation, attention, and residual connections.
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        model_params: dict
    ):
        # Use 'aggregation' (not 'aggregator') for consistency with ImprovedMPNN
        super().__init__(aggr=model_params.get('aggregation', 'mean'))

        hidden_dim = model_params.get('hidden_dim', 128)
        dropout_rate = model_params.get('dropout', 0.5)
        use_batchnorm = model_params.get('use_batchnorm', True)
        use_attention = model_params.get('use_attention', False)

        # Number of layers in each MLP
        n_node_layers = model_params.get('node_mlp_layers', 2)
        n_edge_layers = model_params.get('edge_mlp_layers', 2)
        n_msg_layers  = model_params.get('message_mlp_layers', 2)
        n_final_layers= model_params.get('final_mlp_layers', 2)
        use_residual = model_params.get('use_residual', False)

        # Node MLP (default 2 layers — same structure as ImprovedMPNN)
        self.node_mlp = build_mlp(
            input_dim=num_node_features,
            hidden_dim=hidden_dim,
            n_layers=n_node_layers,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            is_final_mlp=False,  # Outputs hidden_dim
            use_residual=use_residual
        )

        # Edge MLP
        self.edge_mlp = build_mlp(
            input_dim=num_edge_features,
            hidden_dim=hidden_dim,
            n_layers=n_edge_layers,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            is_final_mlp=False,
            use_residual=use_residual
        )

        # Message MLP: combines x_j + edge_attr
        # Input = hidden_dim*2, output = hidden_dim
        self.message_mlp = build_mlp(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            n_layers=n_msg_layers,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            is_final_mlp=False,
            use_residual=use_residual
        )

        # Final MLP: outputs a single scalar per graph
        self.final_mlp = build_mlp(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=n_final_layers,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            is_final_mlp=True,  # Last layer outputs 1 neuron
            use_residual=False
        )

        # Optional attention layer
        if use_attention:
            self.attention_layer = AttentionLayer(hidden_dim)
        else:
            self.attention_layer = None

        # Optional additional residual connection
        self.residual_connection = model_params.get('residual_connection', False)
        if self.residual_connection:
            self.residual_layer = Residual(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

    def forward(self, x, edge_index, edge_attr, batch):
        """
        x:          [num_nodes, num_node_features]
        edge_index: [2, num_edges]
        edge_attr:  [num_edges, num_edge_features]
        batch:      [num_nodes] — assigns each node to a graph in the batch
        """
        # 1) Process node and edge features
        x = self.node_mlp(x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_mlp(edge_attr)  # [num_edges, hidden_dim]

        # 2) Message passing and aggregation
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [num_nodes, hidden_dim]

        # Optional attention
        if self.attention_layer is not None:
            x = self.attention_layer(x, edge_index, batch)  # [num_nodes, hidden_dim]

        # Optional residual connection
        if self.residual_connection:
            x = self.residual_layer(x)  # [num_nodes, hidden_dim]

        # 3) Global pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        # 4) Final MLP (outputs scalar per graph)
        x = self.final_mlp(x)  # [num_graphs, 1]
        return x

    def message(self, x_j, edge_attr):
        """
        Message based on concatenation of x_j and edge_attr.
        combined = [x_j, edge_attr] -> message_mlp -> [num_edges, hidden_dim]
        """
        combined = torch.cat([x_j, edge_attr], dim=1)  # [num_edges, hidden_dim*2]
        return self.message_mlp(combined)  # [num_edges, hidden_dim]
