import numpy as np
import torch
from torch.nn import Module
import copy
from sklearn.metrics import average_precision_score, roc_auc_score





def hitsk(model, test_data_split, k):
    test_data_split_pos = copy.copy(test_data_split)
    test_data_split_pos.edge_label_index = test_data_split.pos_edge_label_index

    test_data_split_neg = copy.copy(test_data_split)
    test_data_split_neg.edge_label_index = test_data_split.neg_edge_label_index

    return compute_hitsk(model(test_data_split_pos).x, model(test_data_split_neg).x, k )

def compute_hitsk(y_pred_pos, y_pred_neg, k):
    tot = (y_pred_pos > torch.sort(y_pred_neg, descending = True)[0][k]).sum()
    return tot / y_pred_pos.size(0)


def aucroc(logits, ground_truths):
    return roc_auc_score(ground_truths.cpu(), logits.cpu())

def average_precision(logits, ground_truths):
    return average_precision_score(ground_truths.cpu().detach().numpy(), logits.cpu().detach().numpy()) 


def auc_loss(logits, ground_truths):
    return 1. - aucroc(logits, ground_truths)

def ap_loss(logits, ground_truths):
    return 1. -  average_precision(logits, ground_truths)

def losses_sum_closure(losses):
    
    return lambda logits, ground_truths: np.sum([loss(logits, ground_truths) for loss in losses]) 

class StandardLossWrapper(Module):
    def __init__(self, norm,  loss):
        super().__init__()
        self.loss = loss
        self.norm = norm

    def forward(self, batch, ground_truth):
        return self.norm * self.loss(batch.x, ground_truth)

