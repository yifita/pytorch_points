import torch
import numpy as np
import os
import warnings
from collections import OrderedDict

saved_variables = {}
def save_grad(name):
    def hook(grad):
        saved_variables[name] = grad
    return hook

def check_values(tensor):
    """return true if tensor doesn't contain NaN or Inf"""
    return not (torch.any(torch.isnan(tensor)).item() or torch.any(torch.isinf(tensor)).item())

def linear_loss_weight(nepoch, epoch, max, init=0):
    """
    linearly vary scalar during training
    """
    return (max - init)/nepoch *epoch + init


def clamp_gradient(model, clip):
    for p in model.parameters():
        torch.nn.utils.clip_grad_value_(p, clip)

def clamp_gradient_norm(model, max_norm, norm_type=2):
    for p in model.parameters():
        torch.nn.utils.clip_grad_norm_(p, max_norm, norm_type=2)


def weights_init(m):
    """
    initialize the weighs of the network for Convolutional layers and batchnorm layers
    """
    if isinstance(m, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)

def save_network(net, directory, network_label, epoch_label=None, **kwargs):
    """
    save model to directory with name {network_label}_{epoch_label}.pth
    Args:
        net: pytorch model
        directory: output directory
        network_label: str
        epoch_label: convertible to str
        kwargs: additional value to be included
    """
    save_filename = "_".join((network_label, str(epoch_label))) + ".pth"
    save_path = os.path.join(directory, save_filename)
    merge_states = OrderedDict()
    merge_states['states'] = net.cpu().state_dict()
    for k in kwargs:
        merge_states[k] = kwargs[k]
    torch.save(merge_states, save_path)
    net = net.cuda()


def load_network(net, path):
    """
    load network parameters whose name exists in the pth file.
    return:
        INT trained step
    """
    if path[-3:] == "pth":
        loaded_state = torch.load(path)
    else:
        loaded_state = np.load(path).item()
    loaded_param_names = set(loaded_state["states"].keys())
    network = net.module if isinstance(
        net, torch.nn.DataParallel) else net

    # allow loaded states to contain keys that don't exist in current model
    # by trimming these keys;
    own_state = network.state_dict()
    extra = loaded_param_names - set(own_state.keys())
    if len(extra) > 0:
        print('Dropping ' + str(extra) + ' from loaded states')
    for k in extra:
        del loaded_state["states"][k]

    try:
        network.load_state_dict(loaded_state["states"])
    except KeyError as e:
        print(e)
        return 0
    else:
        print('Loaded network parameters from {}'.format(path))
        if "step" in loaded_state:
            return loaded_state["step"]
        else:
            return 0


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def tolerating_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = [x for x in filter(lambda x: x is not None, batch)]
    return torch.utils.data.dataloader.default_collate(batch)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
