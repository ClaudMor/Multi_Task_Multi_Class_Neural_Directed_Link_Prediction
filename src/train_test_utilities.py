import torch
import numpy as np
import copy
import time
import gc



def train(train_data, model, train_loss_fn, optimizer,device, num_epochs, lrscheduler = None, early_stopping = False, val_loss_fn = None, val_datasets = None, val_loss_aggregation = "sum", validation_on_device = True, patience = None, use_sparse_representation = False, retrain_data = None, epoch_print_freq = 10): # , train_idxs = None, val_idxs = None, retrain_idxs = None
    
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

        optimizer.zero_grad(set_to_none=True)
        pred = model(train_data)

        loss = train_loss_fn(pred, y_true)

        pred = None
        gc.collect()
        torch.cuda.empty_cache()

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



def compute_loss_on_validation(val_data, model, val_loss_fn,  validation_on_device, device, use_sparse_representation = False, eval = True):
    if eval:
        model.eval()

    if not validation_on_device:
        model.cpu()

    with torch.no_grad():

        y_true = val_data.edge_label

        val_pred = model(val_data).x

        # ic(val_pred)
        val_loss = val_loss_fn(val_pred.reshape(-1),y_true.reshape(-1))

    if eval:
        model.train()

    if not validation_on_device:
        model.to(device)

    return val_loss
    



@torch.no_grad()
def evaluate_link_prediction(model, test_data, metrics_dict, test_data_on_device = False, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ):

    if not test_data_on_device:
        model.cpu()

    model.eval()
    

    logits_test_data = model(test_data).x.cpu()

    out_dict = {}
    for metric_name, metric in metrics_dict.items():
        out_dict[metric_name] = metric(logits_test_data.reshape(-1).detach(), test_data.edge_label.reshape(-1).detach())


    if not test_data_on_device:
        model = model.to(device)

    return out_dict
