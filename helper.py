import torch
import json
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
#from ptflops import get_model_complexity_info
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.utils.prune as prune
from model import densenet121


class CIFAR10Dataset(Dataset):
    """CIFAR10 Dataset

    Parameters
    ----------
    root = str
        Directory where the data is located or downloaded to
    
    train = bool
        If True training set it returned, if False test set it returned

    Attributes
    ----------
    dataset = CIFAR10
        instance of torchvision CIFAR10 class
    """
    def __init__(self, root, train = True, download = True):
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ]
            )
        else: transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ]
            )

        self.dataset = CIFAR10(
            root = root,
            train = train,
            download = download,
            transform = transform,
        )
    
    def __len__(self):
        """return length of dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get selected sample
        
        Parameters
        ----------
        idx = int
            Index of sample to get

        Returns
        -------
        x = torch.Tensor
            Selected sample at idx
        y = torch.Tensor
            Target label at idx
        """
        return self.dataset[idx]
    
def get_layers(model):
    """Get all layers which are to prune and all layers in general from a given model
    
    Parameters
    ----------
    model = nn.Module
        given model to get layers from

    Returns
    -------
    layers_to_prune = dict
        dictionary with names and instances of all layers to prune

    all_layers = dict
        dictionary with names and instances of all layers
    """
    layers_to_prune = {}
    all_layers = {}
    for name, module in model.named_modules():

        all_layers[name] = module
        
        # only prune convolutional and linear layers and save names of pruned layers
        if isinstance(module, nn.Conv2d) | isinstance(module, nn.Linear):
            layers_to_prune[name] = module
    return layers_to_prune, all_layers

def prune_net(sparsity = 0.8, module_dict = {}, param = 'weight'):
    """Prune all layers included in module_dict dictionary
    
    Parameters
    ----------
    resnet = ResNet
        given resnet model instance

    sparsity = float or list
        if float value between 0.0 and 1.0 giving the percentage of sparsity to reach in all layers
        if list provides different pruning ratios for each layer
    
    module_list = list
        list of all layers to prune

    method = str {'LASSO', 'gmp', 'random', 'greedy_layer'}
        Pruning method to use

    to_prune = str {'weight', 'bias'} (only weight was used for my experiment)
        which parameter to be pruned
    """
    parameters_to_prune = ()

    for layer in module_dict:
        parameters_to_prune += ((module_dict[layer], param),)
    # apply L1Unstructured pruning globally
    prune.global_unstructured(parameters_to_prune, prune.L1Unstructured, amount = sparsity)

def create_ticket(trained_model, initial_model_dict):
    """Creates the lottery ticket from a trained and pruned model

    Parameters
    ----------
    trained_model = nn.Module
        trained and pruned model to use as the "mask" for the ticket

    initial_state_dict = state dict
        state dict with weight values from initialization

    model_size = nn.Module
        variable holding the size of all the models, (resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202)

    Returns
    -------
    ticket = nn.Module
        computed ticket with weight values of initialisation but with the same weights removed as the trained and pruned model
    """
    print(f"\nCreating ticket...\n")
    ticket = densenet121()

    ticket.load_state_dict(initial_model_dict)

    pruned_layers, _ = get_layers(trained_model)
    unpruned_layers, _ = get_layers(ticket)

    for pruned, unpruned in zip(pruned_layers, unpruned_layers):
        prune.custom_from_mask(unpruned_layers[unpruned].cpu(), "weight", pruned_layers[pruned].weight_mask.cpu())
    return ticket