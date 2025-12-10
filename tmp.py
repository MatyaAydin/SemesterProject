# ---------- GNN with three branches (temp, press, flow) ----------
class GNN(nn.Module):
    def __init__(self,
                 num_nodes: int = 48,
                 hidden_size = 16,
                 rnn_layers: int = 2,
                 horizon: int = 1,
                 graph_k: int = 8,
                 graph_tau: float = 1):

        super().__init__()
        self.horizon = horizon

        
        self.encoder = nn.Linear(1, hidden_size)

        self.node_emb = NodeEmbedding(num_nodes, hidden_size)
        self.temporal = RNN(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 n_layers=rnn_layers, cell='gru',
                                 return_only_last_state=True)
        self.graph_layer = DifferentiableKnnGraphLayer(num_nodes, k = graph_k, tau = graph_tau)
        self.spatial = DiffConv(in_channels=hidden_size, out_channels=hidden_size, k = 1)

        self.decoder = nn.Linear(hidden_size, horizon)
        

        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x):
        """
        x: [batch, time, num_nodes]
        returns: x_horizon [batch, horizon, num_nodes, 1]
        """
        batch, time, num_nodes = x.shape


        # encode
        x = x.unsqueeze(-1)              # [batch, time, num_nodes, 1]
        x_enc = self.encoder(x)               # [batch, time, num_nodes, hidden_size]

        
        # add node embedding (broadcast)
        emb = self.node_emb().view(1, 1, num_nodes, -1)  # [1,1,num_nodes,hidden]
        x_emb = x_enc + emb                          # [batch,time,num_nodes,hidden]

        # temporal encoder: returns [batch * num_nodes, hidden] per tsl.RNN behavior
        
        h_temp = self.temporal(x_emb)  # [batch * num_nodes, hidden]
        #h_temp = x_emb[:,-1,:,:]
        #h_temp = x_emb.mean(dim = 1)
        

        # Graph learning: get edge_index & edge_weight (shared across batch)
        edge_index, edge_weight = self.graph_layer(emb, None)
       


        # Pass to spatial conv. GATConv in tsl accepts (x, edge_index, edge_weight=...)
        
        spatial_out = self.spatial(h_temp, edge_index, edge_weight = edge_weight)
        # spatial_out typically is (features, attn). Extract features
        if isinstance(spatial_out, tuple):
            h_spatial = spatial_out[0]
        else:
            h_spatial = spatial_out


        x1 = F.elu(h_spatial)

        # Decode to forecasts
        x_out = self.decoder(x1)  # [batch * num_nodes, horizon]

        x_horizon = self.rearrange(x_out)
        return x_horizon, edge_index, edge_weight