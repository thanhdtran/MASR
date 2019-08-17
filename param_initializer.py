#https://github.com/kevinzakka/pytorch-goodies
import math
import torch

def uniform_initialization(params, stdv=0.1):
    '''
    stdv normally is calculated by 1/math.sqrt(hidden_size)

    initialize weights with uniform distribution in range [-stdv, stdv]
    :param params: a pytorch Parameters like nn.Parameter or module or Embedding layer
    :return:
    '''
    return params.data.uniform_(-stdv, stdv)
def zero_initialization(params):
    '''
    fill the params with all 0 (similar to Zeros initializer in Keras)
    :param params:
    :return:
    '''
    return params.data.fill_(0)

def constant_initialization(params, c):
    '''
    fill params with all values set to c (it is Constant initializer in Keras)
    :param params:
    :param c:
    :return:
    '''
    return params.data.fill_(c)

def normal_initialization(params, stdv=0.1):
    '''
    Gau initializer with normal distribution
    :param params:
    :param stdv:
    :return:
    '''
    return params.data.normal_(0, stdv)

def lecun_uniform_initialization(params, fan_in=0):
    '''
    draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(3 / fan_in)
    where  fan_in is the number of input units in the weight tensor.
    :param params:
    :param fan_in: is the number of input units in the weight tensor
    :return:
    '''
    if fan_in == 0:
        fan_in = params.size()[-1]
    limit = math.sqrt(3. / fan_in)
    return uniform_initialization(params, limit)

def he_normal_initialization(params):
    '''
    draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in)
    where  fan_in is the number of input units in the weight tensor.
    :param params:
    :param fan_in:
    :return:
    '''
    torch.nn.init.kaiming_normal_(params.weight, mode='fan_in')

def he_uniform_initialization(params):
    return torch.nn.init.kaiming_uniform(params.weight, mode='fan_in')

def xavier_normal_initialization(params):
    return torch.nn.init.xavier_normal_(params)

def selu_initialization(m):
    '''
    init a model with selu parameters
    :param m: is a module.
    :return:
    '''
    if isinstance(m, torch.nn.Conv2d):
        fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        torch.nn.init.normal(m.weight, 0, math.sqrt(1. / fan_in))
    elif isinstance(m, torch.nn.Linear):
        fan_in = m.in_features
        torch.nn.init.normal(m.weight, 0, math.sqrt(1. / fan_in))


def selu_init_model(model):
    '''
    init the whole model with selu initialization
    :param model:
    :return:
    '''
    for m in model.modules():
        selu_initialization(m)

def batch_norm_model(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant(m.weight, 1)
            torch.nn.init.constant(m.bias, 0)

def orthogonal_init_model(model):
    '''
    Orthogonality is a desirable quality in NN weights in part because it is norm preserving,
    i.e. it rotates the input matrix, but cannot change its norm (scale/shear).
    This property is valuable in deep or recurrent networks,
    where repeated matrix multiplication can result in signals vanishing or exploding.
    :param model:
    :return:
    '''
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.orthogonal(m.weight)

