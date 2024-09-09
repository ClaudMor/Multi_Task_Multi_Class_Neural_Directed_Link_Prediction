
############ FROM HERE #####################
import sys
import copy
sys.path.insert(0, './src')

import numpy as np
import pandas as pd


import torch
from torch.nn import BCEWithLogitsLoss, NLLLoss
from torch_geometric import seed_everything

import gc

import methods



import sys 
from IPython.core.ultratb import ColorTB
sys.excepthook = ColorTB()

seed_everything(12345)

# Please set the parameters below
dataset = "google" # one of "cora","citeseer","google"
training_framework = "multiclass" # One of "baseline", "multiclass","scalarization","multiobjective"
model_name = "magnet"
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Detects gpu if available
num_runs = 5     # Number of dataset splits to average over


# Then run the entire script
if training_framework == "multiclass":
    model_name += "_multiclass"

num_epochs                = methods.setup_suggested_parameters_sets[dataset][model_name]["num_epochs"]
val_loss_fn               = methods.setup_suggested_parameters_sets[dataset][model_name]["val_loss_fn"]                      # The validation loss for early stopping. Default is the sum of AUC and AP over the validation set.
early_stopping            = methods.setup_suggested_parameters_sets[dataset][model_name]["early_stopping"]                   # True or False
use_sparse_representation = methods.models_suggested_parameters_sets[dataset][model_name]["use_sparse_representation"]        # True or False
optimizer_params          = methods.setup_suggested_parameters_sets[dataset][model_name]["optimizer_params"]
add_remaining_self_loops_supervision = True 

# only useful for training_framework="multiclass"
remaining_supervision_self_loops = "negatives"

# set node features
if model_name in ["magnet_multiclass","magnet"]:
    features_type = "in_out_deg" 
else: 
    features_type = "OHE"


# get model
model = methods.get_model(dataset,  model_name, device)

# Since all models are lazy, we run a forward to get an initial state dict
_, train_data_directional, _, _, _, _, _, _, _ = methods.get_split_3_tasks_scipy(dataset, features_type,  add_remaining_self_loops_supervision, use_sparse_representation, True, device)
with torch.no_grad():
    _ = model(train_data_directional)


# Get initial state dict in order to always have the same initial configuration 
initial_model_state_dict = copy.deepcopy(model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

# Clear memory
_, train_data_directional, = (None, None)
torch.cuda.empty_cache()


# Array of test set performances and metrics
preds_general = []
preds_directional = []
preds_bidirectional = []
metrics_dict = {"aucroc":  methods.aucroc, "ap": methods.average_precision}

for i in range(num_runs):

    train_data_general, train_data_directional, train_data_bidirectional, val_data_general, val_data_directional, val_data_bidirectional, test_data_general, test_data_directional, test_data_bidirectional = (None, None, None, None, None, None, None, None, None)
    torch.cuda.empty_cache()

    train_data_general, train_data_directional, train_data_bidirectional, val_data_general, val_data_directional, val_data_bidirectional, test_data_general, test_data_directional, test_data_bidirectional = methods.get_split_3_tasks_scipy(dataset, features_type,  add_remaining_self_loops_supervision, use_sparse_representation, True, device)

    if training_framework == "multiclass":
        train_data_general.edge_label = methods.get_multicass_lp_edge_label_from_sparse_adjt(train_data_general, "train", remaining_supervision_self_loops, device).to(device)
        train_data_directional,  train_data_bidirectional = None,None
        gc.collect()
        torch.cuda.empty_cache()


    torch.cuda.empty_cache()

    if training_framework == "multiclass":
        norm, loss_weights_train = methods.compute_weights_multiclass_classification(train_data_general, remaining_supervision_self_loops)
        torch.cuda.empty_cache()
        train_loss  = methods.StandardLossWrapper(norm ,NLLLoss(weight = loss_weights_train, ignore_index = 4))
    else:
        norm, pos_weight = methods.compute_weights_binary_classification(train_data_general)
        torch.cuda.empty_cache()
        if training_framework in ["scalarization", "multiobjective"]:
            norm_bidirectional, pos_weight_bidirectional = methods.compute_weights_binary_classification(train_data_bidirectional)
            train_loss_general = methods.StandardLossWrapper(norm,BCEWithLogitsLoss(pos_weight = pos_weight))
            train_loss_directional = methods.StandardLossWrapper(1.,BCEWithLogitsLoss())
            train_loss_bidirectional = methods.StandardLossWrapper(norm_bidirectional, BCEWithLogitsLoss(pos_weight = pos_weight_bidirectional))
        elif training_framework == "baseline":
            train_loss =methods.StandardLossWrapper(norm,BCEWithLogitsLoss(pos_weight = pos_weight))


    model.load_state_dict(initial_model_state_dict)
    optimizer = optimizer.__class__(model.parameters(), **optimizer.defaults)

    if training_framework == "multiclass":
        methods.train(train_data_general, model,  train_loss , optimizer, device, num_epochs,  early_stopping = True, val_datasets = (val_data_general, val_data_directional, val_data_bidirectional),  val_loss_fn =  val_loss_fn, validation_on_device=True,  patience = 30, retrain_data = None, use_sparse_representation = use_sparse_representation,  epoch_print_freq = 10)
    elif training_framework == "baseline":
        methods.train(train_data_general, model,  train_loss , optimizer, device, num_epochs,  early_stopping = True, val_datasets = (val_data_general,),  val_loss_fn =  val_loss_fn, validation_on_device=True, patience = 200, retrain_data = None, use_sparse_representation = use_sparse_representation,  epoch_print_freq = 10)
    elif training_framework == "scalarization":
        methods.train_3_tasks(train_data_general, train_data_directional, train_data_bidirectional,  model,  train_loss_general, train_loss_directional, train_loss_bidirectional, optimizer, device, num_epochs, lrscheduler = None, early_stopping = True, val_datasets = (val_data_general, val_data_directional, val_data_bidirectional),  val_loss_fn =  val_loss_fn, patience = 200, retrain_data = None, use_sparse_representation = use_sparse_representation,  epoch_print_freq = 10)
    elif training_framework == "multiobjective":
        methods.train_3_tasks_multiobjective(train_data_general, train_data_directional, train_data_bidirectional, model,  train_loss_general, train_loss_directional, train_loss_bidirectional, optimizer, device, num_epochs, lrscheduler = None, early_stopping = True, val_datasets = (val_data_general, val_data_directional, val_data_bidirectional),  val_loss_fn =  val_loss_fn, validation_on_device=True, patience = 200, retrain_data = None, use_sparse_representation = use_sparse_representation,  epoch_print_freq = 10)


    preds_general.append(methods.evaluate_link_prediction(model, test_data_general, metrics_dict = metrics_dict, device = device))
    preds_directional.append(methods.evaluate_link_prediction(model, test_data_directional, metrics_dict = metrics_dict, device = device))
    preds_bidirectional.append(methods.evaluate_link_prediction(model, test_data_bidirectional, metrics_dict = metrics_dict, device = device))


# Print test-set performances
latex_strings = [f"{model_name}".replace("_","-")]
for task_name, preds in zip(("general", "directional", "bidirectional"), (preds_general, preds_directional, preds_bidirectional)):
    print(task_name)
    mean_std_dict = methods.summarize_link_prediction_evaluation(preds)
    latex_string,markdown_table = methods.pretty_print_link_performance_evaluation(mean_std_dict, model_name)
    latex_strings.append(latex_string)
    print(markdown_table)
print("".join(latex_strings + [" \\\\"]))


# Save performances to dataframe
df = pd.DataFrame(data = np.array([[pred["aucroc"] for pred in preds_general], [pred["ap"] for pred in preds_general],
                            [pred["aucroc"] for pred in preds_directional], [pred["ap"] for pred in preds_directional],
                            [pred["aucroc"] for pred in preds_bidirectional], [pred["ap"] for pred in preds_bidirectional]]).T, columns = ["G_ROC_AUC", "G_AP_AUC","D_ROC_AUC","D_AP_AUC","B_ROC_AUC","B_AP_AUC"])

df.to_csv("./results/"+training_framework+"_"+dataset+"_"+model_name+".csv")


# Clear memory on exit
train_data_general, train_data_directional, train_data_bidirectional, val_data_general, val_data_directional, val_data_bidirectional, test_data_general, test_data_directional, test_data_bidirectional = (None, None, None, None, None, None, None, None, None)
torch.cuda.empty_cache()

