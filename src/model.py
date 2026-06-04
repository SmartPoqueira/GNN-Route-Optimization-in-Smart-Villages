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
    Construye un MLP secuencial con 'n_layers' bloques de la forma:
      [Linear -> ReLU -> (BatchNorm1d) -> Dropout]
    
    - Si is_final_mlp=True, la última capa generada será: Linear(hidden_dim, 1)
      sin ReLU/BN/Dropout posterior.
    - Si use_residual=True, se añade una conexión residual entre la entrada y la salida del MLP.
    
    Ejemplos:
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
        # Añade una capa residual si las dimensiones coinciden
        return Residual(mlp)
    else:
        return mlp

class Residual(nn.Module):
    """
    Capa Residual que suma la entrada a la salida de una subcapa.
    """
    def __init__(self, sublayer):
        super(Residual, self).__init__()
        self.sublayer = sublayer
    
    def forward(self, x):
        return x + self.sublayer(x)

class AttentionLayer(nn.Module):
    """
    Capa de Atención que pondera las características de los nodos.
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
        # Calcula las puntuaciones de atención para cada nodo
        attn_scores = self.attention(x)  # [num_nodes, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)  # [num_nodes, 1]
        return x * attn_weights  # [num_nodes, hidden_dim]

class ImprovedMPNN_NAS(MessagePassing):
    """
    Versión flexible de Improved MPNN que permite definir dinámicamente
    la estructura (número de capas) en cada MLP interno: node_mlp, edge_mlp, message_mlp, final_mlp.
    Además, se varían parámetros como hidden_dim, dropout, aggregation, etc.
    Incluye capas avanzadas como atención y conexiones residuales.
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        model_params: dict
    ):
        # IMPORTANTE: usar 'aggregation' en lugar de 'aggregator' para
        # mantener la consistencia con ImprovedMPNN
        super().__init__(aggr=model_params.get('aggregation', 'mean'))
    
        hidden_dim = model_params.get('hidden_dim', 128)
        dropout_rate = model_params.get('dropout', 0.5)
        use_batchnorm = model_params.get('use_batchnorm', True)
        use_attention = model_params.get('use_attention', False)
    
        # Número de capas en cada MLP
        n_node_layers = model_params.get('node_mlp_layers', 2)
        n_edge_layers = model_params.get('edge_mlp_layers', 2)
        n_msg_layers  = model_params.get('message_mlp_layers', 2)
        n_final_layers= model_params.get('final_mlp_layers', 2)
        use_residual = model_params.get('use_residual', False)
    
        # MLP de nodos (por defecto 2 capas -> estructura igual a ImprovedMPNN)
        self.node_mlp = build_mlp(
            input_dim=num_node_features,
            hidden_dim=hidden_dim,
            n_layers=n_node_layers,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            is_final_mlp=False,  # Este MLP produce hidden_dim
            use_residual=use_residual
        )
    
        # MLP de aristas (igual)
        self.edge_mlp = build_mlp(
            input_dim=num_edge_features,
            hidden_dim=hidden_dim,
            n_layers=n_edge_layers,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            is_final_mlp=False,
            use_residual=use_residual
        )
    
        # MLP para combinar x_j + edge_attr en message
        # En ImprovedMPNN: 2 capas con input = hidden_dim*2, salida = hidden_dim
        self.message_mlp = build_mlp(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            n_layers=n_msg_layers,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            is_final_mlp=False,
            use_residual=use_residual
        )
    
        # MLP final que termina en 1 neurona
        self.final_mlp = build_mlp(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=n_final_layers,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            is_final_mlp=True,  # La última capa sale en 1 neurona
            use_residual=False
        )
    
        # Capa de Atención (opcional)
        if use_attention:
            self.attention_layer = AttentionLayer(hidden_dim)
        else:
            self.attention_layer = None
    
        # Opcional: Capa de Conexiones Residuales adicionales
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
        batch:      [num_nodes] (asigna cada nodo a un grafo en el batch)
        """
        # 1) Procesamos los nodos y las aristas
        x = self.node_mlp(x)  # [num_nodes, hidden_dim]
        edge_attr = self.edge_mlp(edge_attr)  # [num_edges, hidden_dim]
    
        # 2) Mensajes y agregación
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [num_nodes, hidden_dim]
    
        # Opcional: Aplicar capa de atención
        if self.attention_layer is not None:
            x = self.attention_layer(x, edge_index, batch)  # [num_nodes, hidden_dim]
    
        # Opcional: Conexión residual adicional
        if self.residual_connection:
            x = self.residual_layer(x)  # [num_nodes, hidden_dim]
    
        # 3) Global pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
    
        # 4) Pasamos por la MLP final (que ya sale con dimensión 1)
        x = self.final_mlp(x)  # [num_graphs, 1]
        return x
    
    def message(self, x_j, edge_attr):
        """
        Mensaje basado en la concatenación de x_j y edge_attr.
        Estructura:
         combined = [x_j, edge_attr] -> message_mlp -> [num_edges, hidden_dim]
        """
        combined = torch.cat([x_j, edge_attr], dim=1)  # [num_edges, hidden_dim*2]
        return self.message_mlp(combined)  # [num_edges, hidden_dim]

