"""
Loss functions for recommender models.

The pointwise, BPR are a good fit for
implicit feedback models trained through negative sampling.
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def single_pointwise_bceloss(positive_predictions, mask=None, average=False):
    positive_labels = np.ones(positive_predictions.size()).flatten()
    is_cuda = positive_predictions.is_cuda
    if is_cuda:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor).cuda()  # fix expected FloatTensor but got LongTensor
    else:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor)  #fix expected FloatTensor but got LongTensor
    positive_predictions = F.sigmoid(positive_predictions)
    positive_loss = F.binary_cross_entropy(positive_predictions, positive_labels)
    loss = positive_loss
    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()
    if average:
        return loss.mean()
    else:
        return loss.sum()


def pointwise_bceloss(positive_predictions, negative_predictions, mask=None, average=False):
    positive_labels = np.ones(positive_predictions.size()).flatten()
    negative_labels = np.zeros(negative_predictions.size()).flatten()

    is_cuda = positive_predictions.is_cuda
    if is_cuda:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor).cuda()  # fix expected FloatTensor but got LongTensor
        negative_labels = Variable(torch.from_numpy(negative_labels)).type(torch.FloatTensor).cuda()  # fix expected FloatTensor but got LongTensor
    else:
        positive_labels = Variable(torch.from_numpy(positive_labels)).type(torch.FloatTensor)  #fix expected FloatTensor but got LongTensor
        negative_labels = Variable(torch.from_numpy(negative_labels)).type(torch.FloatTensor)  #fix expected FloatTensor but got LongTensor

    positive_predictions = F.sigmoid(positive_predictions)
    negative_predictions = F.sigmoid(negative_predictions)

    positive_loss = F.binary_cross_entropy(positive_predictions, positive_labels)
    negative_loss = F.binary_cross_entropy(negative_predictions, negative_labels)
    loss = positive_loss + negative_loss
    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()
    if average:
        return loss.mean()
    else:
        return loss.sum()
def pointwise_loss(positive_predictions, negative_predictions, mask=None, average=False):
    """
    Logistic loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.
    """

    #old version, without using log loss,
    # positives_loss = (1.0 - F.sigmoid(positive_predictions))
    # negatives_loss = F.sigmoid(negative_predictions)

    #my new version, similar to bce, log loss is better because it is a monoto function
    positives_loss = -torch.log(F.sigmoid(positive_predictions))
    negatives_loss = -torch.log(1.0 - F.sigmoid(negative_predictions))

    loss = (positives_loss + negatives_loss)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    if average:
        return loss.mean()
    else: return loss.sum()


def bpr_loss(positive_predictions, negative_predictions, mask=None, average=False):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.

    Parameters
    ----------

    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.

    Returns
    -------

    loss, float
        The mean value of the loss function.

    References
    ----------

    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """

    #old version, which didn't use log loss.
    # loss = (1.0 - F.sigmoid(positive_predictions -
    #                         negative_predictions))

    #my version, using log loss
    loss = - torch.log(torch.sigmoid
                                 (-negative_predictions + positive_predictions)
                        )



    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    if average:
        return loss.mean()
    else:
        return loss.sum()



