# DiffSTG Review

[Link to original paper](https://arxiv.org/abs/2301.13629)

### Overview 

![alt text](./images/image.png)

DiffSTG is a model that uses both spatial temporal graph neural network to capture spatial and temporal dependencies in the data, but also diffusion models to quantify uncertainty in the prediction. It comes after other ST-based models that would just output a scalar.

### Denoiser

The denoising process is conditioned on the graph and the historical node features:

![alt text](./images/image-1.png)

The denoiser $\epsilon_{\theta}$ (aka UGnet) is inspried from a Unet architecture:

![alt text](./images/image-2.png)

The graph convolution block captures the spatial dependencies, whereas the gated causal convolution blocks capture the temporal dependencies. Positional encoding of the (de)noising step is added for causality.

#### Graph convolution:

#### Equations

![alt text](./images/image-5.png)

By definition of $A_{gcn}$, the aggregation is just a normalized sum over the neighbors

##### Code

see `UGnet.py`

```python
class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks)) # Kernel (learnable)
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1)) # Bias (learnable)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5)) # Initialization
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x: [b, c_in, time, n_nodes]
        # Lk: [3, n_nodes, n_nodes]
        if len(Lk.shape) == 2: # if supports_len == 1:
            Lk=Lk.unsqueeze(0)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta,
                            x_c) + self.b  # [b, c_out, time, n_nodes]
        return torch.relu(x_gc + x)
```

#### Gated causal convolution

##### Equations

It consists of a 1D causal (as it is a time series) convolution with a filter $K$ followed by a gated block to filter useful information and introduce nonlinearity:

![alt text](./images/image-3.png)

![alt text](./images/image-4.png)

##### Code

See `UGnet.py`

```python

class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, : -self.chomp_size]


class TcnBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation_size=1, droupout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.padding = (self.kernel_size - 1) * self.dilation_size

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(3, self.kernel_size), padding=(1, self.padding), dilation=(1, self.dilation_size))

        self.chomp = Chomp(self.padding)
        self.drop =  nn.Dropout(droupout)

        self.net = nn.Sequential(self.conv, self.chomp, self.drop)

        self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else None

    def forward(self, x):
        # x: (B, C_in, V, T) -> (B, C_out, V, T)
        out = self.net(x)
        x_skip = x if self.shortcut is None else self.shortcut(x)

        return out + x_skip

```


### Denoiser architecture in the other paper:

![alt text](./images/image-6.png)

![alt text](./images/image-7.png)

$\rho_m$ is a moving average of the temporal features. $\delta_m$ is a combination of this MA with the positonal encoding of the denoising step. $\mathbf{W}_k$ is a projection to adapt the dimension. From $\delta_m$, we perform parallel feature extraction: dilatedConv for temporal and GatedGConv for spatial. Embedding from the hidden state from the GRU $h_t$ is then added. $z_m$ filters which states of $g_m$ are important.
This is done for $M$ blocks in parallel whose outputs are then aggregated to get $\epsilon_{\theta}$


### Future work

**bf is in section of paper**, simple text is personal ideas

* **Anomaly detection**: by using variance ?
* **Swap vanilla GCN with another type of aggregation**: i.e spectral graph convolution with or without polynomial approximation,...
* Use other type of temporal block (i.e attention) as GRU is usually outdated for most application
* Adapt noise variance based on variance in dataset ?

In general future work is modifying the denoiser architecture
