import copy
import torch
from torch.nn import Linear, LazyLinear
import torch_sparse
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing


# This conv expects self loops to have already been added, and that the rows of the adjm are NOT normalized by the inverse out-degree +1. Such normalization will be done using the "mean" aggregation inherited from MessagePassing The adjm in question should be the transpose of the usual adjm.
class Conv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = "mean")
        self.W = LazyLinear(out_channels, bias = False)

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def message(self, x_j):
        return x_j

    def forward(self, x, edge_index):
        transformed = self.W(x)

        transformed_aggregated_normalized = self.propagate(edge_index, x = transformed)  

        return transformed_aggregated_normalized

    def reset_parameters(self):
        super().reset_parameters()
        self.W.reset_parameters()





class DiGAE(MessagePassing):

    def __init__(self, alpha_init, beta_init, in_channels, hidden_channels, out_channels):
        super().__init__(aggr = "sum")
        self.W_S_0 = Linear(in_channels, hidden_channels, bias = False)
        self.W_S_1 = Linear(hidden_channels, out_channels, bias = False)
        self.W_T_0 = Linear(in_channels, hidden_channels, bias = False)
        self.W_T_1 = Linear(hidden_channels, out_channels, bias = False)
        self.alpha_init = alpha_init
        self.beta_init  = beta_init
        self.alpha = Parameter(torch.tensor(alpha_init), requires_grad = False )
        self.beta = Parameter(torch.tensor(beta_init), requires_grad = False )
        self.in_channels = in_channels


    def message_and_aggregate(self, adj_t, x):

        return torch.cat((torch_sparse.matmul(adj_t.t(), self.W_dir(x[:, self.in_channels: ]), reduce=self.aggr), torch_sparse.matmul(adj_t, self.W_rev(x[:, :self.in_channels]), reduce=self.aggr)), dim = 1 )


    def forward(self, x, edge_index):
  

        adj_t  = copy.deepcopy(edge_index) 


        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.)


        deg_out = torch_sparse.sum(adj_t, dim=1).pow(-self.beta)
        deg_in  = torch_sparse.sum(adj_t.t(), dim=1).pow(-self.alpha)


        deg_out = torch.masked_fill(deg_out, deg_out == float('inf'), 0.)
        deg_in  = torch.masked_fill(deg_in, deg_in   == float('inf'), 0.)


        adj_t.set_value_(deg_out[adj_t.storage.row()] * deg_in[adj_t.storage.col()], layout = "coo")

        Z_S = torch_sparse.matmul(adj_t, 
                                   self.W_T_1(
                                       torch.nn.functional.relu(torch_sparse.matmul(adj_t.t(), self.W_S_0(x)))))

        Z_T = torch_sparse.matmul(adj_t.t(),
                                    self.W_S_1(
                                        torch.nn.functional.relu(torch_sparse.matmul(adj_t, self.W_T_0(x)))))
        


        return torch.cat((Z_S,Z_T), dim = 1 ) 

    def reset_parameters(self):
        super().reset_parameters()
        self.W_dir.reset_parameters()
        self.W_rev.reset_parameters()
        self.alpha.data = torch.tensor([self.alpha_init]).to(self.alpha.data.device)
        self.beta.data = torch.tensor([self.beta_init]).to(self.beta.data.device)