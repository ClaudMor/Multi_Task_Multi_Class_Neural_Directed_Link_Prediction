from typing import Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer
from torch_geometric_signed_directed.nn.directed.MagNetConv import MagNetConv



class MagNet_link_prediction(nn.Module):
    r"""The MagNet model for link prediction from the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.

    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MagNetConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
    """

    def __init__(self, num_features: int, hidden: int = 2, q: float = 0.25, K: int = 2,
                 activation: bool = True, trainable_q: bool = False, layer: int = 2, dropout: float = 0.5, normalization: str = 'sym', cached: bool = False, sparse: bool = False):
        super().__init__()

        chebs = nn.ModuleList()
        chebs.append(MagNetConv(in_channels=num_features, out_channels=hidden, K=K,
                                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached)) #
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MagNetConv(in_channels=hidden, out_channels=hidden, K=K,
                                    q=q, trainable_q=trainable_q, normalization=normalization, cached=cached)) 

        self.Chebs = chebs

        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()


    def forward(self, batch) -> torch.FloatTensor: 
            """
            Making a forward pass of the MagNet node classification model.

            Arg types:
                * real, imag (PyTorch Float Tensor) - Node features.
                * edge_index (PyTorch Long Tensor) - Edge indices.
                * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
                * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            Return types:
                * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
            """
            new_batch = copy.copy(batch)

            real = copy.copy(new_batch.x)
            imag = torch.clone(real)
            for cheb in self.Chebs:
                real, imag = cheb(real, imag, new_batch.edge_index, new_batch.edge_weight)
                if self.activation:
                    real, imag = self.complex_relu(real, imag)


            
            x = torch.cat((real,imag), dim = 1)

            
            if self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)

            new_batch.x = x

            return new_batch