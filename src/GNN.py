import torch_sparse
import torch
import copy
from torch.nn import Module, ModuleList, Parameter, Sequential, Linear, LeakyReLU, Dropout
from torch.nn.functional import dropout, sigmoid
from torch_geometric.utils import add_remaining_self_loops 
from utils import reset_parameters


class LayerWrapper(Module):
    def __init__(self, layer, normalization_before_activation = None, activation = None, normalization_after_activation =None, dropout_p = None, _add_remaining_self_loops = False, uses_sparse_representation = False  ):
        super().__init__()
        self.activation = activation
        self.normalization_before_activation = normalization_before_activation
        self.layer = layer
        self.normalization_after_activation = normalization_after_activation
        self.dropout_p = dropout_p
        self._add_remaining_self_loops = _add_remaining_self_loops
        self.uses_sparse_representation = uses_sparse_representation

    def forward(self, batch):

    
        new_batch = copy.copy(batch)
        if self._add_remaining_self_loops and not self.uses_sparse_representation:
            new_batch.edge_index, _  = add_remaining_self_loops(new_batch.edge_index)
        elif self._add_remaining_self_loops and self.uses_sparse_representation:
            new_batch.edge_index = torch_sparse.fill_diag(new_batch.edge_index, 2)

        if not self.uses_sparse_representation:
            new_batch.x = self.layer(x = new_batch.x, edge_index = new_batch.edge_index)
        else:
            new_batch.x = self.layer(x = new_batch.x, edge_index = new_batch.edge_index)

        if self.normalization_before_activation is not None:
            new_batch.x = self.normalization_before_activation(new_batch.x)
        if self.activation is not None:
            # ic(new_batch.x)
            new_batch.x = self.activation(new_batch.x)
        if self.normalization_after_activation is not None:
            new_batch.x =  self.normalization_after_activation(new_batch.x)
        if self.dropout_p is not None:
            new_batch.x =  dropout(new_batch.x, p=self.dropout_p, training=self.training)


        return new_batch




class GNN_FB(Module):
    def __init__(self, gnn_layers,  preprocessing_layers = [], postprocessing_layers = []):
        super().__init__()

        self.net = torch.nn.Sequential(*preprocessing_layers, *gnn_layers, *postprocessing_layers )

    
    def forward(self, batch):
        return self.net(batch) 



class DecoderDotProduct(Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):

        new_batch   = copy.copy(batch)


        if batch.edge_label_index in ["full_graph", "salha_biased"]:
            
            new_batch.x = torch.matmul(batch.x, batch.x.t()).reshape(-1,1) 

        else:
            new_batch.x = ( batch.x[batch.edge_label_index[0,:],:] * batch.x[batch.edge_label_index[1,:],:]).sum(dim = 1).reshape(-1,1)
        
        return new_batch


class DecoderGravity(Module):
    def __init__(self, l, EPS = 1e-2, CLAMP = None, train_l = True): 
        super().__init__()
        self.l_initialization = l
        self.l = Parameter(torch.tensor([l]), requires_grad = train_l )
        self.EPS = EPS
        self.CLAMP = CLAMP
    def forward(self, batch):

        new_batch   = copy.copy(batch)

        if batch.edge_label_index in ["full_graph", "directional", "bidirectional"]: 
            m_i = new_batch.x[:,-1].reshape(-1,1).expand((-1,new_batch.x.size(0))).t()
            r = new_batch.x[:,:-1]

            norm = (r * r).sum(dim = 1, keepdim = True)
            r1r2 = torch.matmul(r, r.t())

            r2 = norm - 2*r1r2 + norm.t() 

            logr2 = torch.log(r2 + self.EPS)

            if self.CLAMP is not None:
                logr2 = logr2.clamp(min = -self.CLAMP, max = self.CLAMP)
            
            new_batch.x = (m_i -  self.l * logr2).reshape(-1,1)

        else:

            m_j = new_batch.x[new_batch.edge_label_index[1,:],-1]

            diff = new_batch.x[new_batch.edge_label_index[0,:], :-1] - new_batch.x[new_batch.edge_label_index[1,:], :-1]

            r2 = (diff * diff).sum(dim = 1) 
            new_batch.x = (m_j - self.l * torch.log(r2 + self.EPS)).reshape(-1,1)
            
        return new_batch

    def reset_parameters(self):
        self.l.data = torch.tensor([self.l_initialization]).to(self.l.data.device)




class DecoderGravityMulticlass(Module):
    def __init__(self, l, test_val_binary, EPS = 1e-2, CLAMP = None, train_l = True,): 
        super().__init__()
        self.test_val_binary = test_val_binary
        self.l_initialization = l
        self.l = Parameter(torch.tensor([l]), requires_grad = train_l )
        self.EPS = EPS
        self.CLAMP = CLAMP
    def forward(self, batch):

        new_batch   = copy.copy(batch)

        if batch.edge_label_index in ["full_graph", "salha_biased"]  and self.training: 
            m_j = new_batch.x[:,-1].reshape(-1,1).expand((-1,new_batch.x.size(0))).t()
            r = new_batch.x[:,:-1]

            # ||r1 - r2||^2_2 = r1^2 + r2^2 - 2 r1 * r2

            norm = (r * r).sum(dim = 1, keepdim = True)
            r1r2 = torch.matmul(r, r.t())

            r2 = norm - 2*r1r2 + norm.t() 
            logr2 = torch.log(r2 + self.EPS)

            if self.CLAMP is not None:
                logr2 = logr2.clamp(min = -self.CLAMP, max = self.CLAMP)


            s_ij = (m_j -  self.l * logr2).sigmoid()
            s_ji = s_ij.t()

            p_nu = ((1. - s_ij)*s_ji).reshape(-1,1 )
            p_pu = (s_ij*(1.-s_ji)).reshape(-1,1 )
            p_pb = (s_ij*s_ji).reshape(-1,1)
            p_nb = ((1.-s_ij)*(1.-s_ji)).reshape(-1,1)

            probs = torch.cat((p_nb, p_pu, p_pb, p_nu), dim = 1)

            log_probs = torch.log(probs.clamp(min = 1e-10, max = 1.))


            
            new_batch.x = log_probs


        elif torch.is_tensor(batch.edge_label_index) and not self.training and not self.test_val_binary:

            m_j = new_batch.x[new_batch.edge_label_index[1,:],-1]
            m_i = new_batch.x[new_batch.edge_label_index[0,:],-1]

            diff = new_batch.x[new_batch.edge_label_index[0,:], :-1] - new_batch.x[new_batch.edge_label_index[1,:], :-1] # z1 - z2

            r2 = (diff * diff).sum(dim = 1) # || z1 - z2||^2_2

            s_ij = (m_j -  self.l * torch.log(r2 + self.EPS)).sigmoid()
            s_ji = (m_i -  self.l * torch.log(r2 + self.EPS)).sigmoid()


            p_nu = ((1. - s_ij)*s_ji).reshape(-1,1 )
            p_pu = (s_ij*(1.-s_ji)).reshape(-1,1 )
            p_pb = (s_ij*s_ji).reshape(-1,1)
            p_nb = ((1.-s_ij)*(1.-s_ji)).reshape(-1,1)

            probs = torch.cat((p_nb, p_pu, p_pb, p_nu), dim = 1)

            log_probs = torch.log(probs.clamp(min = 1e-10, max = 1.))

            new_batch.x = log_probs



        elif torch.is_tensor(batch.edge_label_index) and not self.training and self.test_val_binary:


            m_j = new_batch.x[new_batch.edge_label_index[1,:],-1]


            diff = new_batch.x[new_batch.edge_label_index[0,:], :-1] - new_batch.x[new_batch.edge_label_index[1,:], :-1] # z1 - z2

            r2 = (diff * diff).sum(dim = 1) # || z1 - z2||^2_2

            new_batch.x = m_j - self.l * torch.log(r2 + self.EPS)
            
        return new_batch

    def reset_parameters(self):
        # super().reset_parameters()
        self.l.data = torch.tensor([self.l_initialization]).to(self.l.data.device)

class DecoderSourceTarget(Module):
    def __init__(self):
        super().__init__()
    def forward(self, batch):

        new_batch   = copy.copy(batch)

        hidden_dimension = batch.x.size(1)
        half_dimension = int(hidden_dimension/2)

        if batch.edge_label_index in ["full_graph", "directional", "bidirectional"] and self.training:

            source = batch.x[:, :half_dimension]
            target = batch.x[:, half_dimension:]

            new_batch.x = torch.matmul(source, target.t()).reshape(-1,1)

        else:

            new_batch.x = (new_batch.x[new_batch.edge_label_index[0,:], :half_dimension] * new_batch.x[new_batch.edge_label_index[1,:], half_dimension:]).sum(dim = 1).sigmoid().reshape(-1,1)
            
        return new_batch


class DecoderSourceTargetMulticlass(Module):
    def __init__(self, test_val_binary, CLAMP = 9): 
        super().__init__()
        self.test_val_binary = test_val_binary
        self.CLAMP = CLAMP
    def forward(self, batch):

        new_batch   = copy.copy(batch)

        hidden_dimension = batch.x.size(1)
        half_dimension = int(hidden_dimension/2)

        if batch.edge_label_index in ["full_graph", "salha_biased"] and self.training:


            source = batch.x[:, :half_dimension]
            target = batch.x[:, half_dimension:]

            s_ij = sigmoid(torch.matmul(source, target.t()))
            s_ji = s_ij.t()

            p_nu = ((1. - s_ij)*s_ji).reshape(-1,1 )
            p_pu = (s_ij*(1.-s_ji)).reshape(-1,1 )
            p_pb = (s_ij*s_ji).reshape(-1,1)
            p_nb = ((1.-s_ij)*(1.-s_ji)).reshape(-1,1)

            probs = torch.cat((p_nb, p_pu, p_pb, p_nu), dim = 1)

            log_probs = torch.log(probs.clamp(min = 1e-10, max = 1.))



            new_batch.x = log_probs 

            


        elif torch.is_tensor(batch.edge_label_index) and not self.training and not self.test_val_binary:

            source_i = new_batch.x[new_batch.edge_label_index[0,:], :half_dimension]
            target_i = new_batch.x[new_batch.edge_label_index[0,:], half_dimension:]

            source_j = new_batch.x[new_batch.edge_label_index[1,:], :half_dimension]
            target_j = new_batch.x[new_batch.edge_label_index[1,:], half_dimension:]

            s_ij = sigmoid((source_i * target_j).sum(dim = 1))
            s_ji = sigmoid((source_j * target_i).sum(dim = 1))

            p_nu = ((1. - s_ij)*s_ji).reshape(-1,1 )
            p_pu = (s_ij*(1.-s_ji)).reshape(-1,1 )
            p_pb = (s_ij*s_ji).reshape(-1,1)
            p_nb = ((1.-s_ij)*(1.-s_ji)).reshape(-1,1)


            probs = torch.cat((p_nb, p_pu, p_pb, p_nu), dim = 1)

            log_probs = torch.log(probs.clamp(min = 1e-10, max = 1.))

            new_batch.x = torch.cat((p_nb, p_pu, p_pb, p_nu), dim = 1)


        # Use for test/val only. It outputs logits as of now (just take the sigmid for probabilities or the log-sigmoid for log-probabilities)           
        elif torch.is_tensor(batch.edge_label_index) and not self.training and self.test_val_binary:

            
            new_batch.x = (new_batch.x[new_batch.edge_label_index[0,:], :half_dimension] * new_batch.x[new_batch.edge_label_index[1,:], half_dimension:]).sum(dim = 1).sigmoid().reshape(-1)


        return new_batch


class MLP_LP(Module):
    def __init__(self, input_dim, output_dim, bias, activation = torch.nn.Identity()):
        super().__init__()

        self.src_linear = Linear(input_dim, output_dim, bias = bias)
        self.dst_linear = Linear(input_dim, output_dim, bias = bias)

        self.output_dim = output_dim
        self.activation = activation

    def forward(self, batch):

        if batch.edge_label_index in ["full_graph"]:

            srcs_trasf = self.src_linear(batch.x).reshape(batch.num_nodes, 1, self.output_dim)

            dsts_trasf = self.dst_linear(batch.x)

            x =  (srcs_trasf + dsts_trasf).reshape(-1, self.output_dim) 
        
        elif torch.is_tensor(batch.edge_label_index): 

            src_logits = self.src_linear(batch.x[batch.edge_label_index[0,:],:])
            dst_logits = self.dst_linear(batch.x[batch.edge_label_index[1,:],:])


            x = src_logits + dst_logits



        batch.x = self.activation(x)

        return batch

    def reset_parameters(self):
        super().reset_parameters()
        reset_parameters(self.src_linear)
        reset_parameters(self.dst_linear)


class DecoderLinear_for_EffectiveLP(Module):
    def __init__(self, input_dim, output_dim, bias, dropout):
        super().__init__()


        self.mlp_lp = MLP_LP(input_dim, output_dim, bias)
        self.dropout = Dropout(dropout)

    def forward(self, batch):

        batch.x = self.dropout(self.mlp_lp(batch).x)

            
        return batch

       
    
    def reset_parameters(self):
        super().reset_parameters()
        self.mlp_lp.reset_parameters()


    

class DecoderLinear_for_EffectiveLP_multiclass(Module):
    def __init__(self, input_dim, output_dim, bias, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.src_linear = Linear(input_dim, output_dim, bias = bias)
        self.dst_linear = Linear(input_dim, output_dim, bias = bias)
        self.dropout = Dropout(dropout)
        self.output_dim = output_dim # it must be 1

    def forward(self, batch):

        new_batch = copy.copy(batch)

        if batch.edge_label_index in ["full_graph"]:



            srcs_trasf = self.src_linear(new_batch.x).reshape(batch.num_nodes, 1, self.output_dim)



            dsts_trasf = self.dst_linear(new_batch.x)


            s_ij = (srcs_trasf + dsts_trasf).squeeze().sigmoid()
            s_ji = s_ij.t()

            p_nu = ((1. - s_ij)*s_ji).reshape(-1,1 )
            p_pu = (s_ij*(1.-s_ji)).reshape(-1,1 )
            p_pb = (s_ij*s_ji).reshape(-1,1)
            p_nb = ((1.-s_ij)*(1.-s_ji)).reshape(-1,1)

            probs = torch.cat((p_nb, p_pu, p_pb, p_nu), dim = 1)

            log_probs = torch.log(probs.clamp(min = 1e-10, max = 1.))



            new_batch.x = self.dropout(log_probs) 



        elif torch.is_tensor(batch.edge_label_index) and not self.training:

            src_logits = self.src_linear(batch.x[batch.edge_label_index[0,:],:])
            dst_logits = self.dst_linear(batch.x[batch.edge_label_index[1,:],:])
            

            probs = (src_logits + dst_logits).sigmoid()



            new_batch.x = probs 
            
        return new_batch

       
    
    def reset_parameters(self):
        super().reset_parameters()
        reset_parameters(self.src_linear)
        reset_parameters(self.dst_linear)

        