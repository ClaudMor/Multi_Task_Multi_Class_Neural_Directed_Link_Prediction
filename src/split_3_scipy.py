import copy
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, add_remaining_self_loops, degree
from torch_geometric.transforms import ToSparseTensor
from sklearn.model_selection import train_test_split
import input_data

from train_test_utilities import compute_loss_on_validation




def get_split_3_tasks_scipy(dataset_name, features_type, add_remaining_self_loops_supervision, use_sparse_representation, validation_on_device, device):

    adj, features = input_data.load_data(dataset_name, "./data/"+dataset_name+".cites")

    # Create datasets
    num_nodes = adj.shape[0]



    positive_unidirectionals_edge_index_t = np.array(adj.multiply(adj.T == 0).nonzero()).T
    print(f"adj.nnz = {adj.nnz}")


    num_positives_unidirectionals = positive_unidirectionals_edge_index_t.shape[0]
    num_positives_directional_test =  int(num_positives_unidirectionals * 0.1)
    num_positives_directional_val =  int(num_positives_unidirectionals * 0.05)
    train_val_directional_positives, test_directional_positives = train_test_split(positive_unidirectionals_edge_index_t, test_size = num_positives_directional_test, shuffle = True)
    print(f"positive_unidirectionals_edge_index_t.shape = {positive_unidirectionals_edge_index_t.shape}, train_val_directional_positives.shape = {train_val_directional_positives.shape}, test_directional_positives = {test_directional_positives.shape}")
    train_directional_positives, val_directional_positives = train_test_split(train_val_directional_positives, test_size = num_positives_directional_val, shuffle = True)
    print(f"train_directional_positives.shape = {train_directional_positives.shape}, val_directional_positives = {val_directional_positives.shape}")

    train_directional_edge_label_index = torch.cat((
        torch.tensor(train_directional_positives.T),
        torch.tensor(train_directional_positives.T).flip(dims = (0,))
    ), dim = 1)
    val_directional_edge_label_index = torch.cat((
        torch.tensor(val_directional_positives.T),
        torch.tensor(val_directional_positives.T).flip(dims = (0,))
    ), dim = 1)
    test_directional_edge_label_index = torch.cat((
        torch.tensor(test_directional_positives.T),
        torch.tensor(test_directional_positives.T).flip(dims = (0,))
    ), dim = 1)

    num_positives_directional_train = train_directional_positives.shape[0]
    train_directional_edge_label = torch.cat((
        torch.ones(num_positives_directional_train),
        torch.zeros(num_positives_directional_train)
    ))
    val_directional_edge_label = torch.cat((
        torch.ones(num_positives_directional_val),
        torch.zeros(num_positives_directional_val)
    ))
    test_directional_edge_label = torch.cat((
        torch.ones(num_positives_directional_test),
        torch.zeros(num_positives_directional_test)
    ))


    # bidirectional

    positive_bidirectionals_edge_index = np.array(adj.multiply(adj.T).nonzero())

    positive_bidirectionals_one_way_edge_index_t = positive_bidirectionals_edge_index[:, positive_bidirectionals_edge_index[0,:] > positive_bidirectionals_edge_index[1,:] ].T


    num_positives_bidirectionals = positive_bidirectionals_one_way_edge_index_t.shape[0]


    num_positives_bidirectional_test =  int(num_positives_bidirectionals * 0.3)


    num_positives_bidirectional_val = int(num_positives_bidirectionals * 0.15)




    train_val_bidirectional_positives, test_bidirectional_positives = train_test_split(positive_bidirectionals_one_way_edge_index_t, test_size = num_positives_bidirectional_test, shuffle = True)
    train_bidirectional_positives, val_bidirectional_positives = train_test_split(train_val_bidirectional_positives, test_size = num_positives_bidirectional_val, shuffle = True)

    num_positives_bidirectional_train = train_bidirectional_positives.shape[0]
    train_positive_unidirectionals_edge_index_t_idx_sample = torch.arange(num_positives_directional_train, dtype = torch.float).multinomial(num_positives_bidirectional_train, replacement=False)
    train_bidirectional_negatives =  train_directional_positives[train_positive_unidirectionals_edge_index_t_idx_sample,:]

    sample_positive_unidirectionals_edge_index_t_idx = torch.arange(num_positives_unidirectionals, dtype = torch.float).multinomial(num_positives_bidirectional_val + num_positives_bidirectional_test, replacement=False)
    sample_positive_unidirectionals_edge_index_t =  positive_unidirectionals_edge_index_t[sample_positive_unidirectionals_edge_index_t_idx,:]

    val_bidirectional_negatives = sample_positive_unidirectionals_edge_index_t[:num_positives_bidirectional_val, :]
    test_bidirectional_negatives = sample_positive_unidirectionals_edge_index_t[num_positives_bidirectional_val:, :]


    train_bidirectional_edge_label_index = torch.cat((
        torch.tensor(train_bidirectional_positives.T),
        torch.tensor(train_bidirectional_negatives.T).flip(dims = (0,))
    ), dim = 1)
    val_bidirectional_edge_label_index = torch.cat((
        torch.tensor(val_bidirectional_positives.T),
        torch.tensor(val_bidirectional_negatives.T).flip(dims = (0,))
    ), dim = 1)
    test_bidirectional_edge_label_index = torch.cat((
        torch.tensor(test_bidirectional_positives.T),
        torch.tensor(test_bidirectional_negatives.T).flip(dims = (0,))
    ), dim = 1)

    
    train_bidirectional_edge_label = torch.cat((
        torch.ones(num_positives_bidirectional_train),
        torch.zeros(num_positives_bidirectional_train)
    ))
    val_bidirectional_edge_label = torch.cat((
        torch.ones(num_positives_bidirectional_val),
        torch.zeros(num_positives_bidirectional_val)
    ))
    test_bidirectional_edge_label = torch.cat((
        torch.ones(num_positives_bidirectional_test),
        torch.zeros(num_positives_bidirectional_test)
    ))



    # train edge index
    reciprocals_other_way = torch.tensor(positive_bidirectionals_edge_index[:, positive_bidirectionals_edge_index[0,:] <= positive_bidirectionals_edge_index[1,:] ]) # here we keep self loops 
    train_edge_index = torch.cat((
        torch.tensor(train_directional_positives.T),
        torch.tensor(train_bidirectional_positives.T),
        reciprocals_other_way
    ), dim = 1) 

    if add_remaining_self_loops_supervision:
        train_edge_index, _                     = add_remaining_self_loops(train_edge_index, num_nodes = num_nodes)


    # general 
    train_general_edge_label_index = "full_graph"
    tosparse = ToSparseTensor()
    train_adj_t = tosparse(Data(edge_index = train_edge_index.to(dtype = torch.long), num_nodes = num_nodes)).adj_t.t() # NOT TRANSPOSED
    train_general_edge_label = train_adj_t.to_dense()


    val_test_general_negatives = negative_sampling(torch.tensor(adj.nonzero()), num_nodes = num_nodes, num_neg_samples = num_positives_directional_val + num_positives_directional_test + num_positives_bidirectional_val + num_positives_bidirectional_test)

    val_general_negatives  = val_test_general_negatives[:, :(num_positives_directional_val + num_positives_bidirectional_val)]
    test_general_negatives = val_test_general_negatives[:, (num_positives_directional_val + num_positives_bidirectional_val):]
    
    val_general_edge_label_index = torch.cat((torch.tensor(val_directional_positives.T),  torch.tensor(val_bidirectional_positives.T), val_general_negatives ), dim = 1)
    val_general_edge_label = torch.cat((torch.ones(num_positives_directional_val + num_positives_bidirectional_val), torch.zeros(num_positives_directional_val + num_positives_bidirectional_val)), dim = 0)

    test_general_edge_label_index = torch.cat((torch.tensor(test_directional_positives.T),  torch.tensor(test_bidirectional_positives.T), test_general_negatives), dim = 1)
    test_general_edge_label = torch.cat((torch.ones(num_positives_directional_test + num_positives_bidirectional_test), torch.zeros(num_positives_directional_test + num_positives_bidirectional_test)), dim = 0)


    if features_type == "OHE":
        x = torch.eye(num_nodes)
    elif features_type == "in_out_deg":
        print(f"train_edge_index = {train_edge_index}")
        out_deg = degree(train_edge_index[0,:], num_nodes = num_nodes)
        in_deg = degree(train_edge_index[1,:], num_nodes = num_nodes)
        x = torch.cat((in_deg.reshape(-1,1),out_deg.reshape(-1,1)), dim = 1)
    elif features_type == "original":
        x = features
    
    if use_sparse_representation:
        train_edge_index = train_adj_t
    else:
        train_edge_index = train_edge_index.to(torch.long)


    train_data_general = Data(x = x, edge_label = train_general_edge_label.reshape(-1,1),  edge_label_index=train_general_edge_label_index, edge_index = train_edge_index)
    val_data_general   = Data(x = x, edge_label = val_general_edge_label.reshape(-1,1),  edge_label_index=val_general_edge_label_index, edge_index = train_edge_index)
    test_data_general  = Data(x = x, edge_label = test_general_edge_label.reshape(-1,1),  edge_label_index=test_general_edge_label_index, edge_index = train_edge_index)

    train_data_directional = Data(x = x, edge_label = train_directional_edge_label.reshape(-1,1),  edge_label_index=train_directional_edge_label_index, edge_index = train_edge_index)
    val_data_directional   = Data(x = x, edge_label = val_directional_edge_label.reshape(-1,1),  edge_label_index=val_directional_edge_label_index, edge_index = train_edge_index)
    test_data_directional  = Data(x = x, edge_label = test_directional_edge_label.reshape(-1,1),  edge_label_index=test_directional_edge_label_index, edge_index = train_edge_index)


    train_data_bidirectional = Data(x = x, edge_label = train_bidirectional_edge_label.reshape(-1,1),  edge_label_index=train_bidirectional_edge_label_index, edge_index = train_edge_index)
    val_data_bidirectional   = Data(x = x, edge_label = val_bidirectional_edge_label.reshape(-1,1),  edge_label_index=val_bidirectional_edge_label_index, edge_index = train_edge_index)
    test_data_bidirectional  = Data(x = x, edge_label = test_bidirectional_edge_label.reshape(-1,1),  edge_label_index=test_bidirectional_edge_label_index, edge_index = train_edge_index)


    if not validation_on_device:

        return train_data_general.to(device), train_data_directional.to(device), train_data_bidirectional.to(device), val_data_general, val_data_directional, val_data_bidirectional, test_data_general, test_data_directional, test_data_bidirectional
    else:
        return train_data_general.to(device), train_data_directional.to(device), train_data_bidirectional.to(device), val_data_general.to(device), val_data_directional.to(device), val_data_bidirectional.to(device), test_data_general, test_data_directional, test_data_bidirectional




def train_3_tasks(train_data, train_data_directional, train_data_bidirectional, model, train_loss_fn, train_loss_fn_directional, train_loss_fn_bidirectional, optimizer,device, num_epochs, lrscheduler = None, early_stopping = False, val_loss_fn = None, val_datasets = None, val_loss_aggregation = "sum", patience = None, use_sparse_representation = False, retrain_data = None, epoch_print_freq = 10, validation_on_device = True):
    
    model.train()


    initial_model_state_dict = None 

    if early_stopping:
        initial_model_state_dict = copy.deepcopy(model.state_dict())

    if lrscheduler is not None:
        initial_lrscheduler_state_dict = copy.deepcopy(lrscheduler.state_dict())


    
    ES_counter = 0
    ES_loss_previous_epoch = torch.tensor(0)
    val_losses = []
    train_losses = []

    y_true = train_data.edge_label.to(device)
    val_losses_by_dataset = [1.,1.,1.]
    best_number_of_epochs = None
    if early_stopping and retrain_data is None:
        best_model_dict = initial_model_state_dict
    for i in range(num_epochs):

        optimizer.zero_grad()
        tot_val_loss = sum(val_losses_by_dataset)
        pred_general = model(train_data)
        loss_general = ((val_losses_by_dataset[0] / tot_val_loss)**2) * train_loss_fn(pred_general, train_data.edge_label)

        pred_directional = model(train_data_directional)
        loss_directional = ((val_losses_by_dataset[1] / tot_val_loss)**2) * train_loss_fn_directional(pred_directional, train_data_directional.edge_label)

        pred_bidirectional = model(train_data_bidirectional)
        loss_bidirectional = ((val_losses_by_dataset[2] / tot_val_loss)**2) * train_loss_fn_bidirectional(pred_bidirectional, train_data_bidirectional.edge_label)

        loss = loss_general + loss_directional + loss_bidirectional

        train_losses.append(loss.item())



        # Backpropagation
        loss.backward()
        optimizer.step()


        
        if i % epoch_print_freq == 0:
            loss, current = loss.item(), i
            print(f"loss: {loss:>7f}  epoch = {i+1} / {num_epochs}")

        
        if val_datasets is not None:
            val_losses_by_dataset = []
            for val_dataset in val_datasets:
                if val_dataset.edge_label_index.size(1) != 0:
                    val_losses_by_dataset.append(compute_loss_on_validation(val_dataset,  model, val_loss_fn, validation_on_device, device, use_sparse_representation))



            val_loss = None
            if val_loss_aggregation == "sum":
                val_loss = np.sum(val_losses_by_dataset)




            if i>0 and early_stopping:
                if any(val_loss.item() >= previous_val_loss for previous_val_loss in val_losses): 
                    ES_counter += 1

                else:
                    ES_counter = 0

                    if retrain_data is None:
                        best_model_dict = copy.deepcopy(model.state_dict())

                if ES_counter > patience:
                    best_number_of_epochs = np.argmin(val_losses) + 1
                    print(f"val_losses = {val_losses[-10:]}, val_loss = {val_loss.item()},  ES_counter = {ES_counter} \n BREAKING. The best number of epochs is {best_number_of_epochs}")
                    break

                if i % 10 == 0:
                    print(f"val_losses = {val_losses[-5:]}, val_loss = {val_loss.item()},  ES_counter = {ES_counter}")

            val_losses.append(val_loss.item())


    if early_stopping:
        best_number_of_epochs = np.argmin(val_losses) + 1
        print(f"val_losses = {val_losses[-10:]}, ES_counter = {ES_counter} \n EPOCH LIMIT REACHED \n BREAKING. The best number of epochs is {best_number_of_epochs}")

    
    if early_stopping and retrain_data is None:
        model.load_state_dict(best_model_dict)

    elif early_stopping and retrain_data is not None:

        if best_number_of_epochs is None:
            best_number_of_epochs = np.argmin(val_losses) + 1

        print(f"\nRetraining on {best_number_of_epochs} epochs...\n")

        model.load_state_dict(initial_model_state_dict)
        optimizer = optimizer.__class__(model.parameters(), **optimizer.defaults)
        if lrscheduler is not None:
            lrscheduler.load_state_dict(initial_lrscheduler_state_dict)
            lrscheduler.optimizer = optimizer

        start = time.time()
        train(retrain_data, model, train_loss_fn, optimizer,device, best_number_of_epochs, lrscheduler=lrscheduler, val_datasets = None, val_loss_fn=None, early_stopping = False, use_sparse_representation = use_sparse_representation, epoch_print_freq = epoch_print_freq) 
        
        end = time.time()
        print(f"Training time: {end - start} seconds")
