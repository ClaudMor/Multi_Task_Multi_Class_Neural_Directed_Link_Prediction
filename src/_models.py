from torch.nn import Sequential, ReLU, LeakyReLU
from GNN import LayerWrapper, DecoderGravity, DecoderGravityMulticlass, DecoderSourceTarget, DecoderSourceTargetMulticlass, DecoderDotProduct, DecoderLinear_for_EffectiveLP, DecoderLinear_for_EffectiveLP_multiclass, GNN_FB
from Convolution import Conv, DiGAE
from custom_losses import losses_sum_closure, auc_loss, ap_loss
from MagNet import MagNet_link_prediction


def get_model(dataset, model_name, device):
    model = None
    if model_name == "gae":
        model = get_gae(**models_suggested_parameters_sets[dataset][model_name])
    if model_name == "gravity_gae":
        model = get_gravity_gae(**models_suggested_parameters_sets[dataset][model_name])
    elif model_name == "sourcetarget_gae":
        model = get_sourcetarget_gae(**models_suggested_parameters_sets[dataset][model_name])
    elif model_name == "gravity_gae_multiclass":
        model = get_gravity_gae_multiclass(**models_suggested_parameters_sets[dataset][model_name])
    elif model_name == "sourcetarget_gae_multiclass":
        model = get_sourcetarget_gae_multiclass(**models_suggested_parameters_sets[dataset][model_name])
    elif model_name == "mlp_gae_multiclass":
        model = get_mlp_gae_multiclass(**models_suggested_parameters_sets[dataset][model_name], device = device)
    elif model_name == "mlp_gae":
        model = get_mlp_gae(**models_suggested_parameters_sets[dataset][model_name], device = device)
    elif model_name == "digae":
        model = get_digae(**models_suggested_parameters_sets[dataset][model_name], device = device)
    elif model_name == "digae_multiclass":
        model = get_digae_multiclass(**models_suggested_parameters_sets[dataset][model_name], device = device)
    elif model_name == "magnet" or model_name == "magnet_ohe":
        model = get_magnet(**models_suggested_parameters_sets[dataset][model_name], device = device)
    elif model_name == "magnet_multiclass" or model_name == "magnet_multiclass_ohe":
        model = get_magnet_multiclass(**models_suggested_parameters_sets[dataset][model_name], device = device)

    
    return model.to(device)



def get_sourcetarget_gae_multiclass(input_dimension, hidden_dimension, output_dimension, use_sparse_representation):

    
    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },
                
                        ]



    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwarg) for unwrapped_layers_kwarg in unwrapped_layers_kwargs])
    decoder = DecoderSourceTargetMulticlass(test_val_binary = True)
    return Sequential(encoder, decoder)




def get_gravity_gae(input_dimension, hidden_dimension, output_dimension, use_sparse_representation, CLAMP, l , train_l):

    
    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension + 1), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]), LayerWrapper(**unwrapped_layers_kwargs[1])])
    decoder = DecoderGravity(l = l, train_l=train_l, CLAMP = CLAMP)
    return Sequential(encoder, decoder)



def get_gravity_gae_multiclass(input_dimension, hidden_dimension, output_dimension, use_sparse_representation, CLAMP, l , train_l):

    
    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension + 1), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]), LayerWrapper(**unwrapped_layers_kwargs[1])])
    decoder = DecoderGravityMulticlass(l = l, train_l=train_l, CLAMP = CLAMP, test_val_binary = True)
    return Sequential(encoder, decoder)



def get_sourcetarget_gae(input_dimension, hidden_dimension, output_dimension, use_sparse_representation):

    
    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]), LayerWrapper(**unwrapped_layers_kwargs[1])])
    decoder = DecoderSourceTarget()
    return Sequential(encoder, decoder)


def get_gae(input_dimension, hidden_dimension, output_dimension, use_sparse_representation):

    
    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": ReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension + 1), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]), LayerWrapper(**unwrapped_layers_kwargs[1])])
    decoder = DecoderDotProduct()
    return Sequential(encoder, decoder)


def get_mlp_gae_multiclass(input_dimension, hidden_dimension, output_dimension, bias_decoder, use_sparse_representation, dropout, device):

    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": LeakyReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension), 
                        "normalization_before_activation": None, 
                        "activation": LeakyReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]), LayerWrapper(**unwrapped_layers_kwargs[1])])
    decoder = DecoderLinear_for_EffectiveLP_multiclass(output_dimension, 1, bias = bias_decoder, dropout = dropout)
    return Sequential(encoder, decoder)



def get_mlp_gae(input_dimension, hidden_dimension, output_dimension, use_sparse_representation, bias_decoder,dropout, device):

    unwrapped_layers_kwargs = [
                        {"layer":Conv(input_dimension, hidden_dimension), 
                        "normalization_before_activation": None, 
                        "activation": LeakyReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        {"layer":Conv(hidden_dimension, output_dimension), 
                        "normalization_before_activation": None, 
                        "activation": LeakyReLU(), 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]), LayerWrapper(**unwrapped_layers_kwargs[1])])
    decoder = DecoderLinear_for_EffectiveLP(output_dimension, 1, bias = bias_decoder, dropout = dropout) 
    return Sequential(encoder, decoder)




def get_digae(input_dimension, hidden_dimension, output_dimension, alpha_init, beta_init, use_sparse_representation, device, test_val_binary = True):

    unwrapped_layers_kwargs = [
                        {"layer":DiGAE( alpha_init, beta_init, input_dimension, hidden_dimension, output_dimension), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        }]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]),])
    decoder = DecoderSourceTarget()
    return Sequential(encoder, decoder)


def get_digae_multiclass(input_dimension, hidden_dimension, output_dimension, alpha_init, beta_init, use_sparse_representation, device, test_val_binary = True):

    unwrapped_layers_kwargs = [
                        {"layer":DiGAE( alpha_init, beta_init, input_dimension, hidden_dimension, output_dimension), 
                        "normalization_before_activation": None, 
                        "activation": None, 
                        "normalization_after_activation": None, 
                        "dropout_p": None, 
                        "_add_remaining_self_loops": False, 
                        "uses_sparse_representation": use_sparse_representation,
                        },

                        ]


    encoder = GNN_FB(gnn_layers = [ LayerWrapper(**unwrapped_layers_kwargs[0]),])
    decoder = DecoderSourceTargetMulticlass(test_val_binary = test_val_binary)
    return Sequential( encoder, decoder)



def get_magnet(input_dimension, hidden_dimension, q, K, activation, num_layers, trainable_q, dropout, cached, bias_decoder, use_sparse_representation, device):

    encoder = MagNet_link_prediction(num_features = input_dimension,  hidden = hidden_dimension, q=q, K = K, activation = activation, trainable_q = trainable_q,  layer=num_layers, dropout = dropout, normalization = "sym", cached =cached, ) # sparse = use_sparse_representation
    decoder = DecoderLinear_for_EffectiveLP(2*hidden_dimension, 1, bias = bias_decoder, dropout = dropout) 

    return Sequential(encoder, decoder).to(device)
    


def get_magnet_multiclass(input_dimension, hidden_dimension, q, K, activation, num_layers, trainable_q, dropout, cached, bias_decoder, use_sparse_representation, device):

    encoder = MagNet_link_prediction(num_features = input_dimension,  hidden = hidden_dimension, q=q, K = K, activation = activation, trainable_q = trainable_q,  layer=num_layers, dropout = dropout, normalization = "sym", cached =cached, sparse = use_sparse_representation) 
    decoder = DecoderLinear_for_EffectiveLP_multiclass(2*hidden_dimension, 1, bias = bias_decoder, dropout = dropout) 

    return Sequential(encoder, decoder).to(device)

    


models_suggested_parameters_sets = {"cora":{

                                            "gae": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True},

                                            "gravity_gae": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True, "CLAMP" :None, "l": 1. , "train_l":True},
                                           
                                            "gravity_gae_multiclass": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True, "CLAMP" :None, "l": 1. , "train_l":True},

                                            "sourcetarget_gae": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True},

                                            "sourcetarget_gae_multiclass": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True}, 

                                            "mlp_gae_multiclass": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32,  "bias_decoder": False, "dropout":0.5, "use_sparse_representation": True},

                                            "mlp_gae": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32, "bias_decoder": True, "dropout": 0.5, "use_sparse_representation": True},

                                            "digae": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32, "alpha_init":0.5, "beta_init":0.5, "use_sparse_representation": True, "test_val_binary": True},

                                            "digae_multiclass": {"input_dimension":2708 , "hidden_dimension": 64, "output_dimension":32, "alpha_init":0.5, "beta_init":0.5, "use_sparse_representation": True, "test_val_binary": True},

                                            "magnet": {"input_dimension":2 , "hidden_dimension": 16, "q":0.05, "K":2, "activation":True, "num_layers":2, "trainable_q":False, "dropout": 0.5, "cached": False, "use_sparse_representation":False, "bias_decoder":True}, 


                                            "magnet_multiclass": {"input_dimension":2 , "hidden_dimension": 16, "q":0.05, "K":2, "activation":True, "num_layers":2, "trainable_q":False, "dropout": 0.5, "cached": False, "use_sparse_representation":False, "bias_decoder":True}, 
                                         
                                            },
                                        
                                    "citeseer":{

                                            "gae": {"input_dimension":3327 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True},

                                            "gravity_gae": {"input_dimension":3327 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True, "CLAMP" :4, "l": 1. , "train_l":True},

                                            "gravity_gae_multiclass": {"input_dimension":3327 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True, "CLAMP" :None, "l": 1. , "train_l":True},
                                            
                                            "sourcetarget_gae_multiclass": {"input_dimension":3327 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True}, 
                                            

                                            "sourcetarget_gae": {"input_dimension":3327 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True},

                                            "digae_multiclass": {"input_dimension":3327 , "hidden_dimension": 64, "output_dimension":32, "alpha_init":0.5, "beta_init":0.5, "use_sparse_representation": True, "test_val_binary": True},

                                            "mlp_gae_multiclass": {"input_dimension":3327 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True,  "bias_decoder": True, "dropout": 0.5},

                                            "mlp_gae": {"input_dimension":3327 ,"hidden_dimension": 64, "output_dimension":32, "bias_decoder": True, "dropout": 0.5, "use_sparse_representation": True }, 

                                            "digae": {"input_dimension":3327 , "hidden_dimension": 64, "output_dimension":32, "alpha_init":0.5, "beta_init":0.5, "use_sparse_representation": True, "test_val_binary": True},

                                            "magnet": {"input_dimension":2 , "hidden_dimension": 16, "q":0.05, "K":2, "activation":True, "num_layers":2, "trainable_q":False, "dropout": 0.5, "cached": False, "use_sparse_representation":False, "bias_decoder":True}, 

                                            "magnet_multiclass": {"input_dimension":2 , "hidden_dimension": 16, "q":0.05, "K":2, "activation":True, "num_layers":2, "trainable_q":False, "dropout": 0.5, "cached": False, "use_sparse_representation":False, "bias_decoder":True}, 



                                            
                                            },

                                        "google": {
                                            "gae": {"input_dimension":15763 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True},

                                            "gravity_gae": {"input_dimension":15763 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True, "CLAMP" :None, "l": 10. , "train_l":True},

                                            "gravity_gae_multiclass": {"input_dimension":15763 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True, "CLAMP" :None, "l": 10. , "train_l":True},


                                            "sourcetarget_gae": {"input_dimension":15763 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True},

                                            "sourcetarget_gae_multiclass": {"input_dimension":15763 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True}, 

                                            "mlp_gae": {"input_dimension":15763 ,"hidden_dimension": 64, "output_dimension":32, "bias_decoder": True, "dropout": 0.5, "use_sparse_representation": True }, 

                                            "mlp_gae_multiclass": {"input_dimension":15763 , "hidden_dimension": 64, "output_dimension":32, "use_sparse_representation": True,  "bias_decoder": True, "dropout": 0.5},

                                            "digae": {"input_dimension":15763 , "hidden_dimension": 64, "output_dimension":32, "alpha_init":0.5, "beta_init":0.5, "use_sparse_representation": True, "test_val_binary": True},

                                            "digae_multiclass": {"input_dimension":15763 , "hidden_dimension": 64, "output_dimension":32, "alpha_init":0.5, "beta_init":0.5, "use_sparse_representation": True, "test_val_binary": True},


                                            "magnet": {"input_dimension":2 , "hidden_dimension": 16, "q":0.05, "K":2, "activation":True, "num_layers":2, "trainable_q":False, "dropout": 0.5, "cached": False, "use_sparse_representation":True, "bias_decoder":True},  # cannot cache and train q at the same time

                                            "magnet_multiclass": {"input_dimension":2 , "hidden_dimension": 16, "q":0.05, "K":2, "activation":True, "num_layers":2, "trainable_q":False, "dropout": 0.5, "cached": False, "use_sparse_representation":False, "bias_decoder":True}
                                         
                                        }

                                    }

                                        



setup_suggested_parameters_sets = {"cora":{

                                        "gae": {"num_epochs":1000, "optimizer_params":{"lr":0.05}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "sourcetarget_gae_multiclass": {"num_epochs":1000, "optimizer_params":{"lr":1e-2}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "gravity_gae": {"num_epochs":1000, "optimizer_params":{"lr":0.01,}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "gravity_gae_multiclass": {"num_epochs":1000,  "optimizer_params":{"lr":0.01}, "early_stopping":True, "add_remaining_self_loops_supervision" : False, "remaining_supervision_self_loops" : "ignore", "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "sourcetarget_gae": {"num_epochs":1000, "optimizer_params":{"lr":0.01}, "add_remaining_self_loops_supervision" : True, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  }, 

                                        "mlp_gae_multiclass": {"num_epochs":1000, "optimizer_params":{"lr":1e-3}, "add_remaining_self_loops_supervision" : False, "remaining_supervision_self_loops" : "negatives", "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "mlp_gae": {"num_epochs":1000,  "optimizer_params":{"lr":2e-3}, "add_remaining_self_loops_supervision" : False, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "digae": {"num_epochs":1000, "optimizer_params":{"lr":2e-2}, "add_remaining_self_loops_supervision" : False, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "digae_multiclass": {"num_epochs":1000,   "optimizer_params":{"lr":0.002} , "add_remaining_self_loops_supervision" : False, "remaining_supervision_self_loops" : "negatives", "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "magnet": {"num_epochs":3000 , "early_stopping":True, "val_loss_fn":  losses_sum_closure([ap_loss, auc_loss]), "optimizer_params":{"lr":1e-3, "weight_decay":5e-4,}  },



                                        "magnet_multiclass": {"num_epochs":3000 ,"early_stopping":True, "val_loss_fn":  losses_sum_closure([ap_loss, auc_loss]),  "optimizer_params":{"lr":1e-3, "weight_decay":5e-4,}  },




                                        },

                                    "citeseer":{

                                        "gae": {"num_epochs":1000, "optimizer_params":{"lr":0.05}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "gravity_gae": {"num_epochs":1000, "optimizer_params":{"lr":0.05}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "gravity_gae_multiclass": {"num_epochs":1000, "optimizer_params":{"lr":0.01}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "sourcetarget_gae_multiclass": {"num_epochs":1000, "optimizer_params":{"lr":0.01}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },                                       

                                        "sourcetarget_gae": {"num_epochs":1000, "optimizer_params":{"lr":0.02}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "digae_multiclass": {"num_epochs":1000,   "optimizer_params":{"lr":0.002} , "add_remaining_self_loops_supervision" : False, "remaining_supervision_self_loops" : "negatives", "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "mlp_gae_multiclass": {"num_epochs":1000, "optimizer_params":{"lr":0.002}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "mlp_gae": {"num_epochs":1000, "optimizer_params":{"lr":2e-3}, "add_remaining_self_loops_supervision" : True, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                
                                        "digae": {"num_epochs":1000, "optimizer_params":{"lr":2e-2}, "add_remaining_self_loops_supervision" : False, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },


                                        "magnet": {"num_epochs":3000 , "early_stopping":True, "val_loss_fn":  losses_sum_closure([ap_loss, auc_loss]), "optimizer_params":{"lr":1e-3, "weight_decay":5e-4,}  },

                                        "magnet_multiclass": {"num_epochs":3000 ,"early_stopping":True, "val_loss_fn":  losses_sum_closure([ap_loss, auc_loss]),  "optimizer_params":{"lr":1e-3, "weight_decay":5e-4,}  },


                                        },

                                    "google":{

                                        "gae": {"num_epochs":1000,  "optimizer_params":{"lr":0.05}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "gravity_gae": {"num_epochs":1000,  "optimizer_params":{"lr":0.05}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "gravity_gae_multiclass": {"num_epochs":1000, "optimizer_params":{"lr":0.01}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "sourcetarget_gae": {"num_epochs":1000, "optimizer_params":{"lr":0.01},"early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "sourcetarget_gae_multiclass": {"num_epochs":1000, "optimizer_params":{"lr":0.01}, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },


                                        "mlp_gae": {"num_epochs":1000, "optimizer_params":{"lr":2e-3}, "add_remaining_self_loops_supervision" : True, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "mlp_gae_multiclass": {"num_epochs":1000, "optimizer_params":{"lr":0.002}, "add_remaining_self_loops_supervision" : True, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },


                                        "digae": {"num_epochs":1000, "optimizer_params":{"lr":2e-2}, "add_remaining_self_loops_supervision" : False, "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },

                                        "digae_multiclass": {"num_epochs":1000,   "optimizer_params":{"lr":0.002} , "add_remaining_self_loops_supervision" : False, "remaining_supervision_self_loops" : "negatives", "early_stopping":True, "val_loss_fn":  losses_sum_closure([auc_loss, ap_loss])  },


                                        "magnet": {"num_epochs":3000 , "early_stopping":True, "val_loss_fn":  losses_sum_closure([ap_loss, auc_loss]), "optimizer_params":{"lr":1e-3, "weight_decay":5e-4,}  },

                                        "magnet_multiclass": {"num_epochs":3000 ,"early_stopping":True, "val_loss_fn":  losses_sum_closure([ap_loss, auc_loss]),  "optimizer_params":{"lr":1e-3, "weight_decay":5e-4,}  }

                                        
                                    }

                                    
                            }