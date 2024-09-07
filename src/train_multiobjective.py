import copy
import numpy as np
import torch

from train_test_utilities import compute_loss_on_validation
from scipy.optimize import minimize, LinearConstraint



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() ) 


def w_norm(coeffs, grad_general_norm, grad_dir_norm, grad_bidir_norm):

    w = coeffs[0] *grad_general_norm + coeffs[1] * grad_dir_norm + coeffs[2] * grad_bidir_norm
    w_l2_norm = np.linalg.norm(w)
    return (w_l2_norm, (2*w.dot(grad_general_norm), 2*w.dot(grad_dir_norm), 2*w.dot(grad_bidir_norm)))




def train_3_tasks_multiobjective(train_data_general, train_data_directional, train_data_bidirectional, model, train_loss_general_fn, train_loss_fn_directional, train_loss_fn_bidirectional, optimizer,device, num_epochs, lrscheduler = None, early_stopping = False, val_loss_fn = None, validation_on_device = True, val_datasets = None, val_loss_aggregation = "sum", patience = None, use_sparse_representation = False, retrain_data = None, epoch_print_freq = 10): # , train_idxs = None, val_idxs = None, retrain_idxs = None
    
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


    val_losses_by_dataset = [1.,1.,1.]
    best_number_of_epochs = None
    if early_stopping and retrain_data is None:
        best_model_dict = initial_model_state_dict
    for i in range(num_epochs):

        optimizer.zero_grad(set_to_none=True)

        pred_general = model(train_data_general)

        loss_general = train_loss_general_fn(pred_general, train_data_general.edge_label)
        loss_general.backward()

        grad_general = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:

                grad_general += parameter.grad.reshape(-1).tolist()

        grad_general = np.array(grad_general)

        loss_general_item = loss_general.item()
        loss_general = None
        pred_general = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        pred_directional = model(train_data_directional)
        loss_directional = train_loss_fn_directional(pred_directional, train_data_directional.edge_label)
        loss_directional.backward()


        grad_directional = []
        for parameter in model.parameters():
            if parameter.requires_grad:
                grad_directional += parameter.grad.reshape(-1).tolist()


        grad_directional = np.array(grad_directional)

        pred_directional = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        loss_directional_item = loss_directional.item()
        loss_directional = None
        preds_bidirectional = model(train_data_bidirectional)
        loss_bidirectional = train_loss_fn_bidirectional(preds_bidirectional, train_data_bidirectional.edge_label)
        loss_bidirectional.backward()

        grad_bidirectional = []
        for parameter in model.parameters():
            if parameter.requires_grad:
                grad_bidirectional += parameter.grad.reshape(-1).tolist()

        grad_bidirectional = np.array(grad_bidirectional)

        loss_bidirectional_item = loss_bidirectional.item()
        loss_bidirectional = None
        preds_bidirectional = None




        general_length  = np.linalg.norm(grad_general)
        directional_length = np.linalg.norm(grad_directional)
        bidirectional_length = np.linalg.norm(grad_bidirectional)

        if general_length != 0:
            grad_general =  grad_general / general_length
        if directional_length != 0:
            grad_directional =  grad_directional / directional_length
        if bidirectional_length != 0:
            grad_bidirectional =  grad_bidirectional / bidirectional_length

        res = minimize(w_norm, [0.33,0.34, 0.33], args = (grad_general, grad_directional, grad_bidirectional), jac = True, bounds = [(0.,1.), (0.,1.), (0.,1.)], constraints = [LinearConstraint(A = [[1,1,1]], lb = 1., ub = 1.)],   )


        grad_mo = torch.tensor(res.x[0]*grad_general + res.x[1]*grad_directional + res.x[2]*grad_bidirectional).to(device)

        idx = 0
        for name, par in model.named_parameters():
            if par.requires_grad:

                shape = tuple(par.grad.shape)
                tot_len = np.prod(shape).astype(int) # shape[0]*shape[1]
                par.grad = grad_mo[idx:(idx + tot_len)].reshape(shape).to(torch.float)
                
                idx += tot_len


        optimizer.step()

        
        if i % epoch_print_freq == 0:

            train_losses = np.round([loss_general_item, loss_directional_item, loss_bidirectional_item], decimals= 3 )
            print(f"train_losses = {train_losses}, ES_counter = {ES_counter}")

        
        if val_datasets is not None:
            val_losses_by_dataset = []
            for val_dataset in val_datasets:
                if val_dataset.edge_label_index.size(1) != 0:
                    val_losses_by_dataset.append(compute_loss_on_validation(val_dataset,  model, val_loss_fn, validation_on_device, device, use_sparse_representation))


            val_loss = None
            if val_loss_aggregation == "sum":
                val_loss = np.sum(val_losses_by_dataset)


            if i>0 and early_stopping:
                if any(val_loss.item() >= previous_val_loss - 0.0001 for previous_val_loss in val_losses): 
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
        train(retrain_data, model, train_loss_general_fn, optimizer,device, best_number_of_epochs, lrscheduler=lrscheduler, val_datasets = None, val_loss_fn=None, early_stopping = False, use_sparse_representation = use_sparse_representation, epoch_print_freq = epoch_print_freq) 
        
        end = time.time()
        print(f"Training time: {end - start} seconds")
