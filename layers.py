import math
import torch
import numpy as np
import torch.nn as nn
import pytorch_utils as my_utils
from torch.autograd import Variable
import global_constants as gc



#######################################

class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

class ManualEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable of input standard deviation
    """
    # def __init__(self, std):
    #     self.std = std

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 0.01)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

class PositionEmbedding(nn.Embedding):
    def __init__(self,num_embeddings, embedding_dim, padding_idx=gc.PADDING_IDX, left_pad=True):
        '''

        :param num_embeddings: equal to length of the input sequence, (max length in user sequences or max length in
         item sequences
        :param embedding_dim: equal to embedding dim of user or item or word.
        :param padding_idx: normally 0, defined in pytorch_utils.py
        :param left_pad: default is True, meaning that we pad the sentences from the left.
        '''
        super(PositionEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx)
        self._padding_idx = padding_idx
        self._left_pad = left_pad
        self._num_embeddings = num_embeddings #max position, or length of input sequence
        self._embedding_dim = embedding_dim
        self.init_parameters()

    def init_parameters(self):
        '''
        init the parameters in here
        :return:
        '''

        position_enc = np.array([
                                    [pos / np.power(10000, 2 * (j // 2) / self._embedding_dim) for j in range(self._embedding_dim)]
                                    if pos != gc.PADDING_IDX else np.zeros(self._embedding_dim) for pos in range(self._num_embeddings)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        # return my_utils.numpy2tensor(position_enc).type(torch.FloatTensor)

        self.weight.data = my_utils.numpy2tensor(position_enc).type(torch.FloatTensor)

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:
            positions = my_utils.make_positions(input.data, self._padding_idx, self._left_pad)

        return super(PositionEmbedding, self).forward(Variable(positions))


def Linear(in_features, out_features, dropout=0):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) * 1. / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class FasterConvTBC(torch.nn.Module):
    """1D convolution over an input of shape (time x batch x channel)

    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, groups=1):
        super(FasterConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = my_utils._single(kernel_size)
        self.padding = my_utils._single(padding)
        self.groups = groups

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size[0], in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return input.contiguous().conv_tbc(self.weight, self.bias, self.padding[0])

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', padding={padding}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    m = FasterConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)

def ConvBCT(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    """C: channel, or embedding dim, T: number of words in the seq, B: batch size"""
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)
