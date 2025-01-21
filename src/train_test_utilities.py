import torch
from torch_geometric.utils import negative_sampling
import numpy as np
import copy
import time
import gc



def train(train_data, model, train_loss_fn, optimizer,device, num_epochs,
          lrscheduler = None, early_stopping = False, val_loss_fn = None,
          val_datasets = None, val_loss_aggregation = "sum",
          validation_on_device = True, patience = None,
          use_sparse_representation = False, retrain_data = None,
          epoch_print_freq = 10): # , train_idxs = None, val_idxs = None, retrain_idxs = None
    
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

    best_number_of_epochs = None
    if early_stopping and retrain_data is None:
        best_model_dict = initial_model_state_dict
    for i in range(num_epochs):

        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model.encoder(train_data.x, train_data.edge_index)

        if not hasattr(train_data.edge_index, "shape"):
            # edge_index is a SparseTensor
            x_pred = model.decoder(z, train_data.edge_label_index)
            loss = train_loss_fn(x_pred, y_true)

        else:

            1/0
            pos_edge_index = train_data.edge_index
            neg_edge_index = negative_sampling(pos_edge_index, train_data.num_nodes, pos_edge_index.shape[1]*1000)

            pos_pred = model.decoder(z, pos_edge_index)
            neg_pred = model.decoder(z, neg_edge_index)

            x_pred = torch.cat([pos_pred, neg_pred], dim=0)
            y_t = torch.cat([torch.ones(pos_pred.size(0),1), torch.zeros(neg_pred.size(0),1)]).to(device)

            loss = train_loss_fn(x_pred, y_t)


        # Backpropagation
        loss.backward()
        optimizer.step()



        
        if i % epoch_print_freq == 0:
            loss, current = loss.item(), i
            print(f"loss: {loss:>7f}  epoch = {i+1} / {num_epochs}")

        
        if val_datasets is not None:
            val_losses_by_dataset = []
            model.eval()
            z = model.encoder(train_data.x, train_data.edge_index)
            for val_dataset in val_datasets:
                if val_dataset.edge_label_index.size(1) != 0:
                    with torch.no_grad():
                        val_pred = model.decoder(z, val_dataset.edge_label_index)
                        val_loss = val_loss_fn(val_pred.reshape(-1),val_dataset.edge_label.reshape(-1))
                        val_losses_by_dataset.append(val_loss)



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
                    print(f"val_losses = {np.round(val_losses[-10:], decimals= 3 )}, val_loss = {val_loss.item()},  ES_counter = {ES_counter} \n BREAKING. The best number of epochs is {best_number_of_epochs}")
                    break

                if i % 10 == 0:
                    print(f"val_losses = {np.round(val_losses[-10:], decimals= 3 )}, val_loss = {val_loss.item()},  ES_counter = {ES_counter}")

            val_losses.append(val_loss.item())


    if early_stopping:
        best_number_of_epochs = np.argmin(val_losses) + 1
        print(f"val_losses = {np.round(val_losses[-10:], decimals= 3 )}, ES_counter = {ES_counter} \n EPOCH LIMIT REACHED \n BREAKING. The best number of epochs is {best_number_of_epochs}")

    
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


@torch.no_grad()
def evaluate_link_prediction(model, test_data, metrics_dict, test_data_on_device = False, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ):

    if not test_data_on_device:
        model.cpu()

    model.eval()
    

    z = model.encoder(test_data.x, test_data.edge_index)
    logits_test_data = model.decoder(z, test_data.edge_label_index).flatten()
    # logits_test_data = model(test_data).x.cpu()

    out_dict = {}
    for metric_name, metric in metrics_dict.items():
        out_dict[metric_name] = metric(logits_test_data.reshape(-1).detach(), test_data.edge_label.reshape(-1).detach())


    if not test_data_on_device:
        model = model.to(device)

    return out_dict
