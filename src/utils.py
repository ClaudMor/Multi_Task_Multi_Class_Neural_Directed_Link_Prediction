
import numpy as np
from math import log10, floor
from pandas import DataFrame
import torch



# Model utilities
def reset_parameters(module):
    if hasattr(module, 'reset_parameters'):
                print(f"resetting {module}")
                module.reset_parameters()
    for layer in module.children():
            # print(f"layer 1= {layer}")
            if hasattr(layer, 'reset_parameters'):
                print(f"resetting {layer}")
                layer.reset_parameters()
            elif len(list(layer.children())) > 0:
                reset_parameters(layer)


def print_model_parameters_names(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def compute_weights_binary_classification(train_data):
    tot_train_edges = train_data.edge_label.size(0)
    tot_pos_train_edges = int(train_data.edge_label.sum())
    tot_neg_edges_train = tot_train_edges - tot_pos_train_edges
    pos_weight = torch.tensor(tot_neg_edges_train / tot_pos_train_edges)
    norm = tot_train_edges / (tot_neg_edges_train + pos_weight * tot_pos_train_edges)
    return norm, pos_weight


def compute_weights_multiclass_classification(train_data, remaining_supervision_self_loops):
    _, classes_sizes_train = train_data.edge_label.unique(return_counts = True)
    if remaining_supervision_self_loops == "ignore":
        classes_sizes_train = classes_sizes_train[:-1]
    classes_sizes_train_max = torch.max(classes_sizes_train)
    loss_weights_train = classes_sizes_train_max / classes_sizes_train
    norm = classes_sizes_train.sum() / (loss_weights_train * classes_sizes_train).sum()
    return norm, loss_weights_train



def summarize_link_prediction_evaluation(performances):
    mean_std_dict = {}
    metrics_names = performances[0].keys()
    for metric in metrics_names:
        vals = []
        for run in performances:
            vals.append(run[metric])

        mean_std_dict[metric] = {"mean":np.nanmean( list(filter(None, vals)) ), "std": np.nanstd( list(filter(None, vals)) / np.sqrt(len(performances)) )}

    return mean_std_dict



def round_to_first_significative_digit(x):
    digit = -int(floor(log10(abs(x))))
    return digit, round(x, digit)

def pretty_print_link_performance_evaluation(mean_std_dict, model_name):
    performances_strings = {}

    for (metric,mean_std) in mean_std_dict.items():
        if np.isnan(mean_std["mean"]):
            performances_strings[metric] = str(None)
        elif mean_std["std"] == 0:
            digit, mean_rounded = round_to_first_significative_digit(mean_std["mean"]) 
            performances_strings[metric] = str(mean_rounded) + " $\\pm$ " + str(mean_std["std"])
        else:
            digit, std_rounded = round_to_first_significative_digit(mean_std["std"])
            mean_rounded = round(mean_std["mean"], digit)
            performances_strings[metric] = str(mean_rounded) + " $\\pm$ " + str(std_rounded)

    
    df = DataFrame(performances_strings.values(), columns = [model_name.replace("_","-")], index = performances_strings.keys()  )

    latex_string = "".join( [f" & {performances_strings[metric]}" for metric in performances_strings.keys()])

    return latex_string, df.to_markdown(index=True)