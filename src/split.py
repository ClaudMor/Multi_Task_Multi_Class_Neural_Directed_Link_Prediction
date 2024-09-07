# All datasets have been run with the following parameters, except citeseer for which the bidirectional validation set was empty (too few bidirectional edges).
# frac_positive_edges_directional_test = 0.1
# frac_positive_edges_directional_val = 0.05
# frac_positives_bidirectional_test = 0.3
# num_positives_bidirectional_val = 0.15 for all except citeseer for which it was 0
# num_general_val  = num_positives_bidirectional_val + num_positive_edges_directional_val
# num_general_test = num_positives_bidirectional_test + num_positive_edges_directional_test


import copy
import numpy as np
import torch
import torch_sparse
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, remove_self_loops, negative_sampling, to_dense_adj, add_remaining_self_loops, to_undirected, to_edge_index, degree
from torch_geometric.transforms import ToSparseTensor

import input_data


def GetSelfLoops(edge_index):
    # edge_index = copy.deepcopy(_edge_index)
    # removed_self_loops = None
    # if hasattr(data, "edge_index"):
    self_loops = edge_index[:, edge_index[0, :] ==  edge_index[1, :] ]

    return self_loops


def RemoveSelfLoops(_edge_index):
    edge_index = copy.deepcopy(_edge_index)
    # removed_self_loops = None
    # if hasattr(data, "edge_index"):
    removed_self_loops = edge_index[:, edge_index[0, :] ==  edge_index[1, :] ]

    return remove_self_loops(edge_index), removed_self_loops


def RemoveReciprocalEdges(_edge_index, return_removed_reciprocal_edges = False):
    edge_index = copy.copy(_edge_index)
    # Store the edge_index with reciprocal links for later use 
    # original_edge_index = edge_index
    edge_index_symm = to_undirected(edge_index)
    adj_sparse_symm        = to_scipy_sparse_matrix(edge_index_symm) #to_dense_adj(edge_index_symm).squeeze()
    adj_sparse = to_scipy_sparse_matrix(edge_index)
    adj_tilde       =  (adj_sparse_symm - adj_sparse).T
    adj_tilde.eliminate_zeros()
    # Prevent numerical issues
    # adj_tilde[adj_tilde < 0.5] = 0
    # removed_edges = torch.nonzero(adj_symm - ).t()
    edge_index_no_reciprocal, _ = from_scipy_sparse_matrix(adj_tilde) #adj_tilde.indices() #torch.nonzero(adj_tilde).t()
    edge_index = edge_index_no_reciprocal

    if not return_removed_reciprocal_edges:
        return edge_index
    else:
        removed_reciprocals_sp = (adj_sparse - adj_tilde).T
        removed_reciprocals_sp.eliminate_zeros()
        removed_reciprocals, _ = from_scipy_sparse_matrix(removed_reciprocals_sp)
        return edge_index, removed_reciprocals


def get_split(dataset_name, features_type, add_remaining_self_loops_supervision, use_sparse_representation, device):
# Load data

    adj, features = input_data.load_data(dataset_name, "../data/"+dataset_name+".cites")
    edge_index, edge_weights = from_scipy_sparse_matrix(adj)

    # Remove self-loops
    (edge_index_wout_self_loops, _), removed_self_loops = RemoveSelfLoops(edge_index) 

    edge_index_wout_self_loops_st, _, = remove_self_loops(edge_index)

    # Remove reciprocals
    edge_index_wout_reciprocals, reciprocals = RemoveReciprocalEdges(edge_index_wout_self_loops, return_removed_reciprocal_edges = True)
    # Compute number of positives unidirectionals
    num_positives_unidirectionals = edge_index_wout_reciprocals.size(1)


    # Validation Biased
    num_positive_edges_directional_test =  int(num_positives_unidirectionals * 0.1)
    num_positive_edges_directional_val =  int(num_positives_unidirectionals * 0.05)

    # Sample the indices of the unidirectional positives to include in the validation and test sets
    random_unidirectional_sample_idxs = torch.arange(num_positives_unidirectionals, dtype = torch.float).multinomial(num_positive_edges_directional_test + num_positive_edges_directional_val, replacement=False)

    train_mask = torch.full((num_positives_unidirectionals,), True)
    train_mask[random_unidirectional_sample_idxs] = False

    val_test_directional_positives =  edge_index_wout_reciprocals[:,~train_mask]
    val_directional_positives, test_directional_positives = val_test_directional_positives[:, :num_positive_edges_directional_val ], val_test_directional_positives[:, num_positive_edges_directional_val: ] 

    # Remove val/test unidirectional positives from training edge index
    train_edge_index = edge_index_wout_reciprocals[:, train_mask]


    # Split val and test
    val_directional_edge_label_index = torch.cat((val_directional_positives, val_directional_positives[[1,0], :]), dim = 1)
    val_directional_edge_label = torch.cat((torch.ones(num_positive_edges_directional_val), torch.zeros(num_positive_edges_directional_val)), dim = 0 )

    test_directional_edge_label_index = torch.cat((test_directional_positives, test_directional_positives[[1,0], :]), dim = 1)
    test_directional_edge_label = torch.cat((torch.ones(num_positive_edges_directional_test), torch.zeros(num_positive_edges_directional_test)), dim = 0 )




    # bidirectional validation

    # Split bidirectionals in two sets
    reciprocals_one_way = reciprocals[:, reciprocals[0,:] > reciprocals[1,:] ]
    reciprocals_other_way = reciprocals[:, reciprocals[0,:] < reciprocals[1,:] ]



    num_positives_bidirectionals = reciprocals_other_way.size(1)

    num_positives_bidirectional_test =  int(num_positives_bidirectionals * 0.3)
    num_positives_bidirectional_val = None
    if dataset_name == "citeseer":
        num_positives_bidirectional_val = 0
    else:
        num_positives_bidirectional_val =  int(num_positives_bidirectionals * 0.15)


    # Sample the indices of the bidirectional negatives to include in the validation and test sets
    num_positives_unidirectionals = train_edge_index.size(1)
    random_unidirectional_sample_idxs = torch.arange(num_positives_unidirectionals, dtype = torch.float).multinomial(num_positives_bidirectional_test + num_positives_bidirectional_val, replacement=False)

    test_val_bidirectional_negatives = train_edge_index[:, random_unidirectional_sample_idxs][[1,0],:]


    # train, val and test bidirectional positives
    val_bidirectional_positives = reciprocals_other_way[:, :num_positives_bidirectional_val]
    test_bidirectional_positives = reciprocals_other_way[:, num_positives_bidirectional_val:(num_positives_bidirectional_val + num_positives_bidirectional_test)]
    train_bidirectional_positives = reciprocals_other_way[:, (num_positives_bidirectional_val + num_positives_bidirectional_test):]

    # Add one direction of all reciprocals to training set, and also the other direction for training reciprocals. Also re-add the previously removed self-loops, although afterwards we add_remaining_self_loops
    train_edge_index = torch.cat((train_edge_index, reciprocals_one_way, train_bidirectional_positives, removed_self_loops), dim = 1)


    # val and test bidirectionals
    val_bidirectional_edge_label_index =  torch.cat((val_bidirectional_positives, test_val_bidirectional_negatives[:, :num_positives_bidirectional_val]), dim = 1)
    val_bidirectional_edge_label = torch.cat((torch.ones(num_positives_bidirectional_val), torch.zeros(num_positives_bidirectional_val)), dim = 0)

    test_bidirectional_edge_label_index =  torch.cat((test_bidirectional_positives, test_val_bidirectional_negatives[:, num_positives_bidirectional_val:]), dim = 1)
    test_bidirectional_edge_label = torch.cat((torch.ones(num_positives_bidirectional_test), torch.zeros(num_positives_bidirectional_test)), dim = 0)




    # General validation and test
    num_general_val  = num_positives_bidirectional_val + num_positive_edges_directional_val #int(edge_index.size(1) * 0.1)
    num_general_test = num_positives_bidirectional_test + num_positive_edges_directional_test

    val_test_general_negatives = negative_sampling(edge_index, num_neg_samples = num_general_val + num_general_test)

    val_general_negatives = val_test_general_negatives[:, :num_general_val]
    val_general_negatives_edge_label = torch.zeros(num_general_val)

    val_general_positives = torch.cat((val_directional_positives,val_bidirectional_positives), dim = 1)
    val_general_edge_label_index = torch.cat((val_general_positives, val_general_negatives), dim =1 )
    val_general_edge_label = torch.cat((torch.ones(num_general_val), torch.zeros(num_general_val) ), dim = 0 )

    test_general_negatives = val_test_general_negatives[:, num_general_val:]
    test_general_positives = torch.cat((test_directional_positives,test_bidirectional_positives), dim = 1)
    test_general_edge_label_index = torch.cat((test_general_positives, test_general_negatives), dim =1 )
    test_general_edge_label = torch.cat((torch.ones(num_general_test), torch.zeros(num_general_test) ), dim = 0 )



    # combine general, directional and bidirectinal validation edge_index
    val_edge_label_index = torch.cat((val_general_negatives, val_directional_edge_label_index, val_bidirectional_edge_label_index), dim = 1)
    val_edge_label = torch.cat((val_general_negatives_edge_label, val_directional_edge_label, val_bidirectional_edge_label), dim = 0)


    # Create datasets
    num_nodes = edge_index.max() + 1
    if features_type == "OHE":
        x = torch.eye(num_nodes)
    elif features_type == "in_out_deg":
        out_deg = degree(train_edge_index[0,:], num_nodes = num_nodes)
        in_deg = degree(train_edge_index[1,:], num_nodes = num_nodes)
        x = torch.cat((in_deg.reshape(-1,1),out_deg.reshape(-1,1)), dim = 1)
    elif features_type == "original":
        x = features



    if add_remaining_self_loops_supervision:
        train_edge_index, _ = add_remaining_self_loops(train_edge_index, num_nodes = num_nodes)

    tosparse = ToSparseTensor()
    train_adj_t = tosparse(Data(num_nodes = num_nodes, edge_index = train_edge_index)).adj_t.t() # we transpose it after `tosparse` so that the result is NOT transposed (even though we'll call it adj_t...)
    train_edge_label = train_adj_t.to_dense()

    if not use_sparse_representation:
        train_data = Data(x = x, edge_label = train_edge_label.reshape(-1,1),  edge_label_index="full_graph", edge_index = add_remaining_self_loops(train_edge_index, num_nodes = num_nodes)[0])
        val_data   = Data(x = x, edge_label = val_edge_label.reshape(-1,1),  edge_label_index=val_edge_label_index, edge_index = add_remaining_self_loops(train_edge_index, num_nodes = num_nodes)[0])
    else:

        train_data = Data(x = x, edge_label = train_edge_label.reshape(-1,1),  edge_label_index="full_graph",adj_t = torch_sparse.fill_diag(train_adj_t, fill_value = 1.) )
        val_data   = Data(x = x, edge_label = val_edge_label.reshape(-1,1),  edge_label_index=val_edge_label_index,adj_t = torch_sparse.fill_diag(train_adj_t, fill_value = 1.) )



    val_data_general = copy.deepcopy(val_data)
    val_data_general.edge_label_index = val_general_edge_label_index
    val_data_general.edge_label = val_general_edge_label.reshape(-1,1)

    test_data_general = copy.deepcopy(val_data)
    test_data_general.edge_label_index = test_general_edge_label_index
    test_data_general.edge_label = test_general_edge_label.reshape(-1,1)


    val_data_directional = copy.deepcopy(val_data)
    val_data_directional.edge_label_index = val_directional_edge_label_index
    val_data_directional.edge_label = val_directional_edge_label.reshape(-1,1)


    test_data_directional = copy.deepcopy(val_data)
    test_data_directional.edge_label_index = test_directional_edge_label_index
    test_data_directional.edge_label = test_directional_edge_label.reshape(-1,1)


    val_data_bidirectional = copy.deepcopy(val_data)
    val_data_bidirectional.edge_label_index = val_bidirectional_edge_label_index
    val_data_bidirectional.edge_label = val_bidirectional_edge_label.reshape(-1,1)


    test_data_bidirectional = copy.deepcopy(val_data)
    test_data_bidirectional.edge_label_index = test_bidirectional_edge_label_index
    test_data_bidirectional.edge_label = test_bidirectional_edge_label.reshape(-1,1)

    return train_data.to(device), val_data_general, val_data_directional, val_data_bidirectional, test_data_general, test_data_directional, test_data_bidirectional


# # Assumes self-loops have already been added
# def get_multicass_lp_edge_label_from_sparse_adjt(_data, split, remaining_supervision_self_loops, device):
#     # 0. = negative bidirectional
#     # 1. = positives unidirectional
#     # 2. = positives bidirectional
#     # 3. = negatives unidirectional
#     data = copy.deepcopy(_data)
#     if hasattr(data, "adj_t"):
#         edge_index, _ = to_edge_index(data.adj_t)
#     else:
#         edge_index = data.edge_index
#         data.adj_t = ToSparseTensor()(data).adj_t.t()

#     dense_adj = data.adj_t.to_dense()



#     if data.edge_label_index in ["full_graph"] and split == "train":
        
#         # if hasattr(data, "adj_t"):
#         #     edge_index, _ = to_edge_index(data.adj_t)
#         # else:
#         #     edge_index = data.edge_index

#         edge_index_wout_reciprocals, removed_reciprocals = RemoveReciprocalEdges(edge_index, return_removed_reciprocal_edges = True)

#         # removed_reciprocals_wout_self_loops, _ = remove_self_loops(removed_reciprocals)

        

#         if remaining_supervision_self_loops == "ignore":
#             dense_adj.fill_diagonal_(4)
#         elif remaining_supervision_self_loops == "positives":
#             dense_adj.fill_diagonal_(2)
#         elif remaining_supervision_self_loops == "negatives":
#             dense_adj.fill_diagonal_(0)

#         dense_adj[removed_reciprocals[0], removed_reciprocals[1]] = 2

#         dense_adj[edge_index_wout_reciprocals[1], edge_index_wout_reciprocals[0]] = 3

#         return dense_adj.reshape(-1).type(torch.long)



#     # we might use reverses of data.edge_label_index[0] instead of the whole dense_adj to relabel positive edges (it would be more efficient).
#     elif torch.is_tensor(data.edge_label_index) and split in ["val", "test"] :

#         dense_adj[data.edge_label_index[0], data.edge_label_index[1]] = data.edge_label
#         reverses_binary_labels = dense_adj[data.edge_label_index[1], data.edge_label_index[0]]

#         new_edge_labels = []

#         for edge, edge_label, reverse_binary_label in zip(data.edge_label_index.t(), data.edge_label, reverses_binary_labels):
#             if edge[0] == edge[1]:
#                 new_edge_labels.append(4)
#             elif edge_label == 1 and reverse_binary_label == 1 and edge[0] != edge[1]:
#                 new_edge_labels.append(2)
#             elif edge_label == 1 and reverse_binary_label == 0:
#                 new_edge_labels.append(1)
#             elif edge_label == 0 and reverse_binary_label == 1:
#                 new_edge_labels.append(3)
#             elif edge_label == 0 and reverse_binary_label == 0:
#                 new_edge_labels.append(0)
        
#         return torch.tensor(new_edge_labels).reshape(-1).type(torch.long).to(device)



#     return data





# Assumes self-loops have already been added
def get_multicass_lp_edge_label_from_sparse_adjt(_data, split, remaining_supervision_self_loops, device):
    # 0. = negative bidirectional
    # 1. = positives unidirectional
    # 2. = positives bidirectional
    # 3. = negatives unidirectional

    data = copy.deepcopy(_data)

    # if hasattr(data, "adj_t"):
    #     edge_index, _ = to_edge_index(data.adj_t)
    # else:
    #     edge_index = data.edge_index
    #     data.adj_t = ToSparseTensor()(data).adj_t.t()

    # dense_adj = data.adj_t.to_dense()



    # if data.edge_label_index in ["full_graph"] and split == "train":

    dense_adj = data.edge_label.reshape(data.num_nodes, data.num_nodes)

    edge_label_index, _ = to_edge_index(torch_sparse.SparseTensor.from_dense(dense_adj))


    
    # if hasattr(data, "adj_t"):
    #     edge_index, _ = to_edge_index(data.adj_t)
    # else:
    #     edge_index = data.edge_index

    edge_label_index_wout_reciprocals, removed_reciprocals = RemoveReciprocalEdges(edge_label_index, return_removed_reciprocal_edges = True)

    # removed_reciprocals_wout_self_loops, _ = remove_self_loops(removed_reciprocals)

    

    if remaining_supervision_self_loops == "ignore":
        dense_adj.fill_diagonal_(4)
    elif remaining_supervision_self_loops == "positives":
        dense_adj.fill_diagonal_(2)
    elif remaining_supervision_self_loops == "negatives":
        dense_adj.fill_diagonal_(0)

    dense_adj[removed_reciprocals[0], removed_reciprocals[1]] = 2

    dense_adj[edge_label_index_wout_reciprocals[1], edge_label_index_wout_reciprocals[0]] = 3

    return dense_adj.reshape(-1).type(torch.long)

    #     # data.edge_label = dense_adj.reshape(-1).type(torch.long)

    # # we might use reverses of data.edge_label_index[0] instead of the whole dense_adj to relabel positive edges (it would be more efficient).
    # elif torch.is_tensor(data.edge_label_index) and split in ["val", "test"] :

    #     dense_adj[data.edge_label_index[0], data.edge_label_index[1]] = data.edge_label
    #     reverses_binary_labels = dense_adj[data.edge_label_index[1], data.edge_label_index[0]]

    #     new_edge_labels = []

    #     for edge, edge_label, reverse_binary_label in zip(data.edge_label_index.t(), data.edge_label, reverses_binary_labels):
    #         if edge[0] == edge[1]:
    #             new_edge_labels.append(4)
    #         elif edge_label == 1 and reverse_binary_label == 1 and edge[0] != edge[1]:
    #             new_edge_labels.append(2)
    #         elif edge_label == 1 and reverse_binary_label == 0:
    #             new_edge_labels.append(1)
    #         elif edge_label == 0 and reverse_binary_label == 1:
    #             new_edge_labels.append(3)
    #         elif edge_label == 0 and reverse_binary_label == 0:
    #             new_edge_labels.append(0)
        
    #     return torch.tensor(new_edge_labels).reshape(-1).type(torch.long).to(device)

    #     # data.edge_label = torch.tensor(new_edge_labels).reshape(-1).type(torch.long).to(device)



    # return data
