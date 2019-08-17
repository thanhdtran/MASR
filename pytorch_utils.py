import torch
import numpy as np
import collections
from itertools import repeat
import random

def _to_iterable(val, num):

    try:
        iter(val)
        return val
    except TypeError:
        return (val,) * num

def is_cuda(x):
    return x.is_cuda

def flatten(x):
    '''
    flatten high dimensional tensor x into an array
    :param x:
    :return: 1 dimensional tensor
    '''
    dims = x.size()[1:] #remove the first dimension as it is batch dimension
    num_features = 1
    for s in dims: num_features *= s
    return x.contiguous().view(-1, num_features)
def numpy2tensor(x):
    return torch.from_numpy(x)

def tensor2numpy(x):
    # return x.numpy()
    return cpu(x).numpy()

def transfer_to_gpu(x):
    #make sure x is a Variable or a Module
    if torch.cuda.is_available():
        x = x.cuda()
        return x
def transfer_to_cpu(x):
    return x.cpu()

def print_weight(x):
    # make sure x is a layer, for example: convolutional layer or fully connected layer
    print (x.weight.data)

def print_bias(x):
    # make sure x is a layer, for example: convolutional layer or fully connected layer
    print (x.bias.data)

def gpu(tensor, use_cuda = False):
    if use_cuda:
        return tensor.cuda()
    else:
        return tensor
def cpu(tensor):
    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor

#from spotlight
def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 64)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

#from spotlight
def shuffle(*arrays, **kwargs):

    random_state = kwargs.get('random_state')

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    #random_state.shuffle(shuffle_indices) #using this from spotlight will not produce deterministic results
    np.random.shuffle(shuffle_indices) #use this to produce deterministic results

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)

#from spotlight
def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )

#from spotlight
def set_seed(seed, cuda=False):

    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

#convert ids to torch Tensor
def _predict_process_ids(user_ids, item_ids, num_items, use_cuda):

    if item_ids is None:
        item_ids = np.arange(num_items, dtype=np.int64)

    if np.isscalar(user_ids):
        user_ids = np.array(user_ids, dtype=np.int64)

    user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
    item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

    if item_ids.size()[0] != user_ids.size(0):
        # user_ids = user_ids.expand(item_ids.size())
        raise ValueError('User ids and item ids are not same size')

    user_var = gpu(user_ids, use_cuda)
    item_var = gpu(item_ids, use_cuda)

    return user_var.squeeze(), item_var.squeeze()


#use to detach some module, prevent updating gradients.
def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val

#get number of parameters in a model
def get_num_params(model):
    return sum([p.data.nelement() for p in model.parameters()])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

######## TO MAKE POSITION EMBEDDDING ##########################
def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
