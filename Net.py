import param_initializer as initializer
import layers as my_layers
from torch.autograd import Variable
import numpy as np
import pytorch_utils as my_utils
import global_constants as gc
import torch
from torch import nn
import torch.nn.functional as F
from pos_encoding import position_encoding as PE, temporal_encoding as TE
import global_constants as gc


L1, L2 = ('l1', 'l2')
def L2_pow2_func(x):
    #square the L2 distance
    return x **2

def L1_func(x):
    return torch.abs(x)

def get_distance(q, m, A, biases=None, dropout=None, non_linear=None):
    #Mahalanobis distance
    dist = q - m
    dist = dist * A
    # if self._non_linear:  # adding non linear
    #     dist = self._non_linear(dist)
    dist = dropout(dist)
    dist = dist ** 2
    return dist.sum(dim=2) + biases

class ParametersModule(nn.Module):
    def __init__(self,
                 n_users, n_items, n_playlists = None, embedding_size=128,
                 item_seq_size = 5,
                 user_embeddings = None, item_embeddings = None, playlist_embeddings = None,
                 user_noise = None, item_noise = None, item_bias_noise=None, playlist_noise = None,
                 W1 = None, W2 = None, A1 = None, A2=None,
                 W1_noise = None, W2_noise = None, A1_noise = None, A2_noise = None,
                 # output_item_embeddings = None, output_user_embeddings = None,
                 item_temporal_embedding = None,
                 use_temporal_encoding = False,
                 adv=False
                 ):
        super(ParametersModule, self).__init__()
        #define user memory, item memory and item output memory

        self._n_users, self._n_items, self._embedding_size = n_users, n_items, embedding_size
        self._n_playlists = n_playlists

        self._item_seq_size = item_seq_size
        self._use_temporal_encoding = use_temporal_encoding
        #define user (A_u and item (A_i) memory embeddings, as well as output item embedding (C_i)
        self._user_embeddings = (
            user_embeddings or
            nn.Embedding(n_users, embedding_size, padding_idx=gc.PADDING_IDX)
            # my_layers.ManualEmbedding(n_users, embedding_size, padding_idx=gc.PADDING_IDX)
        )
        self._item_embeddings = (
            item_embeddings or
            nn.Embedding(n_items, embedding_size, padding_idx=gc.PADDING_IDX)
            # my_layers.ManualEmbedding(n_users, embedding_size, padding_idx=gc.PADDING_IDX)
        )

        self._item_biases = nn.Embedding(n_items, 1)
        initializer.zero_initialization(self._item_biases.weight)

        # combination of target user u and target item j, and playlist p if possible
        self._W1 = (
            W1 or
            (nn.Linear(2*embedding_size, embedding_size))
            # (nn.Linear(3*embedding_size, embedding_size) if n_playlists else nn.Linear(2*embedding_size, embedding_size))
        )

        # self._A1 = nn.Parameter(torch.rand(embedding_size, 1))
        self._A1 =  A1 or nn.Parameter(torch.ones(embedding_size, 1)) #diagnal A1 matrix = A2 vector


        if self._n_playlists:
            # combination of [u, j] with consumed item i
            self._W2 = (
                W2 or
                (nn.Linear(2*embedding_size, embedding_size))
            )

            # self._A2 = nn.Parameter(torch.rand(embedding_size, 1))
            self._A2 =  A2 or nn.Parameter(torch.ones(embedding_size, 1))  # diagnal A2 matrix = A2 vector


            self._playlist_embeddings = (
                    playlist_embeddings or
                    nn.Embedding(n_playlists, embedding_size, padding_idx=gc.PADDING_IDX)
                # my_layers.ManualEmbedding(n_users, embedding_size, padding_idx=gc.PADDING_IDX)
            )

        self._reset_weight()
        if adv:
            self._adv = adv
            self._create_noise(user_noise=user_noise, item_noise=item_noise, item_bias_noise=item_bias_noise,
                               playlist_noise=playlist_noise,
                               W1_noise=W1_noise, W2_noise=W2_noise, A1_noise=A1_noise, A2_noise=A2_noise)

    def _create_noise(self, user_noise, item_noise, item_bias_noise,
                      playlist_noise, W1_noise, W2_noise, A1_noise, A2_noise):
        self._user_noise = user_noise or nn.Embedding(self._n_users, self._embedding_size, padding_idx=gc.PADDING_IDX)
        self._item_noise = item_noise or nn.Embedding(self._n_items, self._embedding_size, padding_idx=gc.PADDING_IDX)
        self._item_biases_noise = item_bias_noise or nn.Embedding(self._n_items, 1)

        self._W1_noise = W1_noise or nn.Linear(2*self._embedding_size, self._embedding_size)
        self._A1_noise = A1_noise  or nn.Parameter(torch.zeros(self._embedding_size, 1))
        if self._n_playlists:
            self._playlist_noise = playlist_noise or nn.Embedding(self._n_playlists, self._embedding_size, padding_idx=gc.PADDING_IDX)
            self._W2_noise = W2_noise or nn.Linear(2 * self._embedding_size, self._embedding_size)
            self._A2_noise = A2_noise or nn.Parameter(torch.zeros(self._embedding_size, 1))

        self._reset_noise()

        # add to two containers, must match playlist of each component in each container.
        self._params_lst = list()
        self._params_lst.append(self._user_embeddings.weight)
        self._params_lst.append(self._item_embeddings.weight)
        self._params_lst.append(self._item_biases.weight)
        self._params_lst.append(self._W1.weight)
        self._params_lst.append(self._A1)
        if self._n_playlists:
            self._params_lst.append(self._playlist_embeddings.weight)
            self._params_lst.append(self._W2.weight)
            self._params_lst.append(self._A2)

        self._noises_lst = list()
        self._noises_lst.append(self._user_noise.weight)
        self._noises_lst.append(self._item_noise.weight)
        self._noises_lst.append(self._item_biases_noise.weight)
        self._noises_lst.append(self._W1_noise.weight)
        self._noises_lst.append(self._A1_noise)
        if self._n_playlists:
            self._noises_lst.append(self._playlist_noise)
            self._noises_lst.append(self._W2_noise.weight)
            self._noises_lst.append(self._A2_noise)

        pass
    def _reset_noise(self):
        initializer.zero_initialization(self._user_noise.weight)
        initializer.zero_initialization(self._item_noise.weight)
        initializer.zero_initialization(self._item_biases_noise.weight)
        initializer.zero_initialization(self._W1_noise.weight)
        initializer.zero_initialization(self._A1_noise)
        if self._n_playlists:
            initializer.zero_initialization(self._playlist_noise.weight)
            initializer.zero_initialization(self._W2_noise.weight)
            initializer.zero_initialization(self._A2_noise)

        pass

    def _reset_transform_identity(self):
        self._W1.weight.data.copy_(my_utils.numpy2tensor
            (
                np.concatenate(
                    (-np.identity(self._embedding_size),
                    np.identity(self._embedding_size)
                    )
                    , axis=0
                ).T  # initialy, it is subtractions of target user u and target item j, u-j
            )
        )
        if self._n_playlists:
            self._W2.weight.data.copy_(my_utils.numpy2tensor
                (
                    np.concatenate(
                        (-np.identity(self._embedding_size),
                         np.identity(self._embedding_size)
                         )
                        , axis=0
                    ).T  # initialy, it is subtractions of target user u and target item j, u-j
                )
            )


    def _reset_transform(self, type='xavier'):
        if type == 'normal':
            initializer.normal_initialization(self._W1.weight, 1.0 / self._embedding_size)
            if self._n_playlists: initializer.normal_initialization(self._W2.weight, 1.0 / self._embedding_size)
        elif type == 'xavier':
            initializer.xavier_normal_initialization(self._W1.weight)
            if self._n_playlists: initializer.xavier_normal_initialization(self._W2.weight)
        elif type == 'lecun':
            initializer.lecun_uniform_initialization(self._W1.weight)
            if self._n_playlists: initializer.lecun_uniform_initialization(self._W2.weight)
        elif type == 'he-normal':
            initializer.he_normal_initialization(self._W1.weight)
            if self._n_playlists: initializer.he_normal_initialization(self._W2.weight)
        elif type == 'he-uniform':
            initializer.he_uniform_initialization(self._W1.weight)
            if self._n_playlists: initializer.he_uniform_initialization(self._W2.weight)
        elif type == 'identity':
            self._reset_transform_identity()
        else:
            self._reset_transform_identity()

    def _reset_weight(self):
        # self._user_embeddings.weight.data.normal_(0, 0.01)#1.0/self._embedding_size)
        self._user_embeddings.weight.data.normal_(0, 1.0/self._embedding_size)
        self._user_embeddings.weight.data[gc.PADDING_IDX].fill_(0)
        # self._item_embeddings.weight.data.normal_(0, 0.01)#1.0/self._embedding_size) #0.01)
        self._item_embeddings.weight.data.normal_(0, 1.0/self._embedding_size)
        self._item_embeddings.weight.data[gc.PADDING_IDX].fill_(0)

        if self._n_playlists:
            # self._playlist_embeddings.weight.data.normal_(0, 0.01)  # 1.0/self._embedding_size) #0.01)
            self._playlist_embeddings.weight.data.normal_(0, 1.0/self._embedding_size)
            self._playlist_embeddings.weight.data[gc.PADDING_IDX].fill_(0)

        self._reset_transform(gc.INIT_TRANSFORM_TYPE)

    def _embed_input(self, x):
        '''

        :param x: the input or item ids, which are the item ids the users consumed before.
        :return:
        '''
        return self._item_embeddings(x)

    def forward(self, x, u, j, p=None):
        '''

        :param x: input: consumed items, format: batch x consumed_item_ids
        :param u: current user. Format: batch x user_ids
        :param j: next item (target item)
        :param p: playlist/playlist
        :return:
        '''
        if self.use_temporal_encoding:
            position_encoding = PE(
                self._embedding_size,
                self._item_seq_size
            )
            temporal_encoding = TE(self._memory_size, self._embedding_size)

            return (position_encoding * self._embed_input(x)).sum(2) + temporal_encoding
        else:
            if self._n_playlists:
                return self._embed_input(x), self._user_embeddings(u), self._item_embeddings(j), self._item_biases(j), self._playlist_embeddings(p)
            else:
                return self._embed_input(x), self._user_embeddings(u), self._item_embeddings(j), self._item_biases(j)



class OutputModule(nn.Module):

    def __init__(self, embedding_size, distance_type = L2, dropout_prob=0.5,
                 non_linear = None, sum_mapping=True, has_playlist = True):
        super(OutputModule, self).__init__()
        self._dist_func = L2_pow2_func if distance_type == L2 else L1_func
        self._dropout = nn.Dropout(p=dropout_prob)
        self._non_linear = non_linear
        # self._sum_func = nn.Linear(2*embedding_size, 1) if sum_mapping else None
        self._has_playlist = has_playlist
        self._reset_weights()


    def _reset_weights(self):
        # initializer.lecun_uniform_initialization(self._sum_func.weight)
        #self._sum_func.weight.data.copy_(my_utils.numpy2tensor(
        #        np.ones(2*self._embedding_size).T # normaly sum function
        #    )
        #)
        pass



    
    def forward(self, weights, q1, q2, m_c, A1, A2, biases, mask):
        '''

        :param weights: batch x num_consumed_items # batch x seq_len
        :param q1, q2: q1 = W1[u,j], q2 = W2[p,j] #u: target user, p:target playlist/baseket, j: target item # batch x embedding_size
        :param m_c: output embeddings of consumed items, batch x seq_len x embedding_size
        :param C: output memory
        :return:
        '''

        q_us = q1.unsqueeze(1).expand_as(m_c) #batch x embedding_size --> batch x seq_len x embedding_size
        A_us = my_utils.flatten(A1.unsqueeze(0)).unsqueeze(0).expand_as(q_us)  # batch x neighbors x embedding_size
        us_distance = get_distance(q_us, m_c, A_us, biases, dropout=self._dropout)
        us_distance = torch.mul(us_distance, mask)  # make padding items have very low scores.
        distance = us_distance
        if self._has_playlist:
            q_ps = q2.unsqueeze(1).expand_as(m_c)  # batch x neighbors x embedding_size
            A_ps = my_utils.flatten(A2.unsqueeze(0)).unsqueeze(0).expand_as(q_ps)  # batch x neighbors x embedding_size
            ps_distance = get_distance(q_ps, m_c, A_ps, biases, dropout=self._dropout)
            ps_distance = torch.mul(ps_distance, mask)
            distance += ps_distance

        dist_vec = torch.mul(distance, weights)

        return dist_vec

class MaskedAttention(nn.Module):
    def __init__(self, distance_type=L2, dropout_prob=0.2, non_linear = None, has_playlist=True):
        super(MaskedAttention, self).__init__()
        self._dist_func = L2_pow2_func if distance_type==L2 else L1_func
        self._dropout = nn.Dropout(p=dropout_prob)
        self._non_linear = non_linear
        self._has_playlist = has_playlist

    def forward(self, q, q_p=None, m=None, A1=None, A2=None, biases=None, mask=None):
        '''

        :param q: the query formed by target user and target song: batch x embedding_size
        :param q_p: the query formed by target playlist and target song: batch x embedding_size
        :param m: batch x neighbors x embedding_size: batch x neighbors x embedding_size
        :param A1: metric learning between user and song: embedding_size x 1
        :param A2: metric learning between playlist and song: embedding_size x 1
        :param mask: mask
        :return: softmax(-DISTANCE(W2[m, q])) where q = W1[target_user, target_item] = W1[u, j]
        '''
        q_us = q.unsqueeze(1).expand_as(m)  # batch x neighbors x embedding_size
        A_us = my_utils.flatten(A1.unsqueeze(0)).unsqueeze(0).expand_as(q_us) # batch x neighbors x embedding_size
        us_distance = get_distance(q_us, m, A_us, biases, dropout=self._dropout) #batch x neighbors
        # us_distance = torch.mul(us_distance, mask) #make padding items have very low scores.
        distance = us_distance
        if self._has_playlist:
            q_ps = q_p.unsqueeze(1).expand_as(m)  # batch x neighbors x embedding_size
            A_ps = my_utils.flatten(A2.unsqueeze(0)).unsqueeze(0).expand_as(q_ps)  # batch x neighbors x embedding_size
            ps_distance = get_distance(q_ps, m, A_ps, biases, dropout=self._dropout)
            # ps_distance = torch.mul(ps_distance, mask)
            distance += ps_distance
        distance = torch.mul(distance, mask)  # make padding items have very low scores.
        distance = -distance

        weights = F.softmax(distance, dim=1)
        return weights


class MASS(nn.Module):

    def __init__(self,
                 n_users, n_items, n_playlists=None, embedding_size=8,
                 item_seq_size=5,
                 distance_type = 'l1',
                 nonlinear_func = 'none',
                 dropout_prob = 0.2,
                 ret_sum = True,
                 adv = False, eps = 1.0,
                 ):
        super(MASS, self).__init__()

        self._n_users, self._n_items, self._embedding_size = n_users, n_items, embedding_size
        self._n_playlists = n_playlists
        self._item_seq_size = item_seq_size
        self._dropout_prob = dropout_prob
        self._ret_sum = ret_sum #return summation at the end of forward or not

        if nonlinear_func == 'relu': self._non_linear = F.relu
        elif nonlinear_func == 'tanh': self._non_linear = torch.tanh
        else: self._non_linear = None

        self._outputModule = OutputModule(embedding_size, distance_type=distance_type, dropout_prob=dropout_prob,
                                          non_linear=self._non_linear,
                                          has_playlist=True if n_playlists else False)

        self._attModule = MaskedAttention(distance_type=distance_type, dropout_prob=dropout_prob,
                                          non_linear=self._non_linear,
                                          has_playlist=True if n_playlists else False)

        ###############create memories:####################
        ### Memory for attention
        self._A_memory = ParametersModule(n_users, n_items, n_playlists,
                                          embedding_size, item_seq_size,
                                          user_embeddings=None, item_embeddings=None, playlist_embeddings=None,
                                          W1 = None, W2 = None,
                                          item_temporal_embedding=None, use_temporal_encoding=False,
                                          adv = adv,
        )

        ####### Memory for making input, output
        self._C_memory = ParametersModule(n_users, n_items, n_playlists,
                                          embedding_size, item_seq_size,
                                          user_embeddings=None, item_embeddings=None, playlist_embeddings=None,
                                          W1 = None, W2 = None,
                                          item_temporal_embedding=None, use_temporal_encoding=False,
                                          adv=adv,
        )

        self._reset_weights()

        if adv:
            self._adv = adv
            self._eps = eps


    def _reset_weights(self):
        pass


    def _make_query(self, u_embed, j_embed, W1=None, W1_noise=None):
        '''

        :param u_embed: target user embed
        :param j_embed: target item embed
        :param p_embed: target playlist/basket embed
        :param W1:
        :return:
        '''
        # transform target user u and target item j
        q = torch.cat([u_embed, j_embed], dim=1)
        q_transform = W1(q) + W1_noise(q) if W1_noise else W1(q)
        q = self._non_linear(q_transform) if self._non_linear else q_transform
        #return nn.Dropout(p=self._dropout_prob)(q)
        return q

    def _make_output_query(self, u_embed, j_embed, W1=None, W1_noise=None):
        # transform target user u and target item j
        q = torch.cat([u_embed, j_embed], dim=1)
        q_transform = W1(q) + W1_noise(q) if W1_noise else W1(q)
        q = self._non_linear(q_transform) if self._non_linear else q_transform
        #return nn.Dropout(p=self._dropout_prob)(q)
        return q

    def _make_output_mask(self, x):
        mask = np.asarray(my_utils.tensor2numpy(x.data.cpu().clone()), dtype=np.float64)
        # mask = my_utils.tensor2numpy(x != 0)
        mask[mask != gc.PADDING_IDX] = 1.0
        # mask[mask <= 0] = float('inf')
        mask[mask <= 0] = 0

        return my_utils.gpu(Variable(my_utils.numpy2tensor(mask)).type(torch.FloatTensor), use_cuda=my_utils.is_cuda(x))

    def _make_mask(self, x):
        mask = np.asarray(my_utils.tensor2numpy(x.data.cpu().clone()), dtype=np.float64)
        # mask = my_utils.tensor2numpy(x != 0)
        mask[mask != gc.PADDING_IDX] = 1.0
        # mask[mask <= 0] = float('inf')
        mask[mask <= 0] = 65535

        return my_utils.gpu(Variable(my_utils.numpy2tensor(mask)).type(torch.FloatTensor), use_cuda=my_utils.is_cuda(x))


    def _get_l2_loss(self):
        item_reg_l2 = torch.norm(self.C_memories[-1]._item_embeddings.weight)
        user_reg_l2 = torch.norm(self.C_memories[-1]._user_embeddings.weight)
        reg = item_reg_l2 + user_reg_l2
        if self._n_playlists:
            reg += torch.norm(self.C_memories[-1]._playlist_embeddings.weight)
        # return item_reg + user_reg
        return reg

    def _update_noise_params(self):
        #attention's params
        for i, noise_param in enumerate(self._A_memory._noises_lst):
            param = self._A_memory._params_lst[i]
            param_std = torch.std(param.data) * self._eps
            noise_param.data = F.normalize(param.grad.data, p=2, dim=1) * param_std

        #main params
        for i, noise_param in enumerate(self._C_memory._noises_lst):
            param = self._C_memory._params_lst[i]
            param_std = torch.std(param.data) * self._eps
            noise_param.data = F.normalize(param.grad.data, p=2, dim=1) * param_std

    def _clear_grad(self, adv=True):
        if adv:
            for noise_param in self._A_memory._noises_lst:
                noise_param.grad = None
            for noise_param in self._C_memory._noises_lst:
                noise_param.grad = None
        else:
            for param in self._A_memory._params_lst:
                param.grad = None
            for param in self._C_memory._params_lst:
                param.grad = None

    def forward(self, x, u, j, p=None, adv=False):
        """

        :param x: the consumed items of user u, format: batch_size x 1 x n_items
        :param u: user, format batch_size x 1
        :param j: next item, format: batch_size x 1
        :param p: the basket/playlist contains the target item
        :return:
        """
        mask = self._make_mask(x)

        A = self._A_memory
        C = self._C_memory

        #get user embedding:
        att_u = A._user_embeddings(u) # batch x embedding_size
        att_j = A._item_embeddings(j) # batch x embedding_size
        att_j_biases = A._item_biases(j) #biases

        att_m = A._item_embeddings(x)  # get the item embeddings in input memory, return batch x seq_len x embedding_size

        att_W1 = A._W1
        att_W1_noise = None

        att_A1 = A._A1

        if adv:
            att_u = att_u + A._user_noise(u)
            att_j = att_j + A._item_noise(j)
            att_j_biases = att_j_biases + A._item_biases_noise(j)
            att_m = att_m + A._item_noise(x)
            att_W1_noise = A._W1_noise
            att_A1 = att_A1 + A._A1_noise

        # make query
        att_q_us = self._make_query(att_u, att_j, att_W1, W1_noise=att_W1_noise) #batch x embedding_size
        if self._n_playlists:
            att_p = A._playlist_embeddings(p)  # batch x embedding_size (embeddings of playlists/playlists)
            att_W2 = A._W2
            att_W2_noise = None
            att_A2 = A._A2
            if adv:
                att_p = att_p + A._playlist_noise(p)
                att_A2 = att_A2 + A._A2_noise

            att_q_ps = self._make_query(att_p, att_j, att_W2, att_W2_noise) #combination between playlist and song

        else:
            att_p, att_W2, att_A2 = None, None, None
            att_q_ps = None


        weights = self._attModule(att_q_us, att_q_ps, att_m, att_A1, att_A2, att_j_biases, mask)

        out_j = C._item_embeddings(j)
        out_j_biases = C._item_biases(j)
        out_u = C._user_embeddings(u)
        out_W1 = C._W1
        out_W1_noise = None
        out_m = C._item_embeddings(x)
        out_A1 = C._A1

        if adv:
            out_j = out_j + C._item_noise(j)
            out_j_biases = out_j_biases + C._item_biases_noise(j)
            out_u = out_u + C._user_noise(u)
            # out_W1.weight = out_W1.weight + C._W1_noise.weight
            out_W1_noise = C._W1_noise
            out_m = out_m + C._item_noise(x)
            out_A1 = out_A1 + C._A1_noise


        #make output query
        out_q_us = self._make_output_query(out_u, out_j,
                                           out_W1, out_W1_noise)  # output combination of target user u and target item j

        if self._n_playlists:
            out_p = C._playlist_embeddings(p)
            out_W2 = C._W2
            out_W2_noise = None
            out_A2 = C._A2
            if adv:
                out_p = out_p + C._playlist_noise(p)
                out_W2_noise = C._W2_noise
                out_A2 = out_A2 + C._A2_noise
            out_q_ps = self._make_output_query(out_p, out_j, out_W2, out_W2_noise)  # combination between playlist and song
        else:
            out_q_ps, out_A2, out_W2, out_p = None, None, None, None

        out_mask = self._make_output_mask(x)
        o = self._outputModule(weights, out_q_us, out_q_ps, out_m, out_A1, out_A2, out_j_biases, out_mask)

        if self._ret_sum:
            return -o.sum(dim=1)
        else:
            return o


class MDR(nn.Module):
    '''
    For adversarial learning, performing 2 steps:
    Step 1: adding _create_noise() where we define all noises for model's parameters.
    Step 2: _reset_noise(): initializing noise weights.

    '''
    def __init__(self, n_users, n_items, n_playlists = None, embedding_size=16,
                 distance_type = 'l1', sum_mapping = True, ret_sum = True,
                 nonlinear_func='none', dropout_prob=0.2, num_layers = 1, adv=False, eps=1.0):
        super(MDR, self).__init__()
        self._n_users = n_users
        self._n_items = n_items
        self._n_playlists = n_playlists

        self._n_factors = embedding_size
        self._embedding_size = embedding_size
        self._ret_sum = ret_sum

        self._dropout = nn.Dropout(p=dropout_prob)

        self._user_embeddings = nn.Embedding(n_users, embedding_size)
        self._item_embeddings = nn.Embedding(n_items, embedding_size)
        self._A = nn.Parameter(torch.ones(embedding_size, 1))


        self._item_biases = nn.Embedding(n_items, 1)
        initializer.zero_initialization(self._item_biases.weight)

        if self._n_playlists:
            self._playlist_embeddings = nn.Embedding(n_playlists, embedding_size)
            self._B = nn.Parameter(torch.rand(embedding_size, 1))


        if nonlinear_func == 'relu': self._non_linear = F.relu
        elif nonlinear_func == 'tanh': self._non_linear = torch.tanh
        else: self._non_linear = None

        self._reset_weights()
        if adv:
            self._adv = adv
            self._eps = eps
            self._create_noise()

    def _create_noise(self):
        self._user_noise = nn.Embedding(self._n_users, self._embedding_size)
        self._item_noise = nn.Embedding(self._n_items, self._embedding_size)
        self._A_noise = nn.Parameter(torch.zeros(self._embedding_size, 1))
        self._item_biases_noise = nn.Embedding(self._n_items, 1)
        if self._n_playlists:
            self._playlist_noise = nn.Embedding(self._n_playlists, self._embedding_size)
            self._B_noise = nn.Parameter(torch.zeros(self._embedding_size, 1))
        self._reset_noise()

        #add to two containers, must match playlist of each component in each container.
        self._params_lst = list()
        self._params_lst.append(self._user_embeddings.weight)
        self._params_lst.append(self._item_embeddings.weight)
        self._params_lst.append(self._A)
        self._params_lst.append(self._item_biases.weight)
        if self._n_playlists:
            self._params_lst.append(self._playlist_embeddings.weight)
            self._params_lst.append(self._B)

        self._noises_lst = list()
        self._noises_lst.append(self._user_noise.weight)
        self._noises_lst.append(self._item_noise.weight)
        self._noises_lst.append(self._A_noise)
        self._noises_lst.append(self._item_biases_noise.weight)
        if self._n_playlists:
            self._noises_lst.append(self._playlist_noise.weight)
            self._noises_lst.append(self._B_noise)
        pass
    def _reset_noise(self):
        initializer.zero_initialization(self._user_noise.weight)
        initializer.zero_initialization(self._item_noise.weight)
        initializer.zero_initialization(self._A_noise)
        initializer.zero_initialization(self._item_biases_noise.weight)
        if self._n_playlists:
            initializer.zero_initialization(self._playlist_noise.weight)
            initializer.zero_initialization(self._B_noise)
        pass
    def _update_noise_params(self):
        for i, noise_param in enumerate(self._noises_lst):
            param = self._params_lst[i]
            param_std = torch.std(param.data) * self._eps
            noise_param.data = F.normalize(param.grad.data, p=2, dim =1) * param_std

        pass
    def _clear_grad(self, adv=True):
        if adv:
            for noise_param in self._noises_lst:
                noise_param.grad = None
        else:
            for param in self._params_lst:
                param.grad = None

    def _reset_weights(self):
        self._user_embeddings.weight.data.normal_(0, 1.0 / self._embedding_size)
        # self._user_embeddings.weight.data.normal_(0, 0.01)
        self._user_embeddings.weight.data[gc.PADDING_IDX].fill_(0)

        self._item_embeddings.weight.data.normal_(0, 1.0 / self._embedding_size)
        # self._item_embeddings.weight.data.normal_(0, 0.01)
        self._item_embeddings.weight.data[gc.PADDING_IDX].fill_(0)

        # initializer.xavier_normal_initialization(self._A)

        if self._n_playlists:
            self._playlist_embeddings.weight.data.normal_(0, 1.0 / self._embedding_size)
            # self._playlist_embeddings.weight.data.normal_(0, 0.01)
            self._playlist_embeddings.weight.data[gc.PADDING_IDX].fill_(0)
            # initializer.xavier_normal_initialization(self._B)

    def _get_l2_loss(self):
        item_reg_l2 = torch.norm(self._item_embeddings.weight)
        user_reg_l2 = torch.norm(self._user_embeddings.weight)
        reg = item_reg_l2 + user_reg_l2
        if self._n_playlists:
            playlist_reg_l2 = torch.norm(self._playlist_embeddings.weight)
            reg += playlist_reg_l2
        return reg

    def execute(self, user_embeds, item_embeds, A, item_biases):

        ## mahalanobis without/with dropout
        v = user_embeds - item_embeds
        v = v * my_utils.flatten(A.unsqueeze(0)).expand_as(v)  # diagnal A
        if self._non_linear: #adding non linear
            v = self._non_linear(v)

        v = self._dropout(v)
        v = v * v

        if self._ret_sum:
            return v.sum(dim=1) + item_biases
        else:
            return v


    def forward(self, x=None, uids = None, sids = None, pids = None, adv=False):
        '''
        :param x:
        :param uids: user ids
        :param sids: song ids
        :param pids: playlist ids
        :return:
        '''
        user_embeds = self._user_embeddings(uids)  # first dimension is batch size
        song_embeds = self._item_embeddings(sids)
        song_biases = self._item_biases(sids)
        A = self._A
        if adv and self._adv:
            user_noise = self._user_noise(uids)
            song_noise = self._item_noise(sids)
            song_biases_noise = self._item_biases_noise(sids)
            A_noise = self._A_noise
            # add noise
            user_embeds = user_embeds + user_noise
            song_embeds = song_embeds + song_noise
            song_biases = song_biases + song_biases_noise
            A = A + A_noise



        ############## MEASURE distance between user and item
        user_embeds = my_utils.flatten(user_embeds)
        song_embeds = my_utils.flatten(song_embeds)
        song_biases = song_biases.squeeze()

        user_song_distance = self.execute(user_embeds, song_embeds, A, song_biases)

        final_distance = user_song_distance
        #####################################################
        ########### MEASURE distance between playlist and item
        if self._n_playlists:
            playlist_embeds = self._playlist_embeddings(pids)
            B = self._B
            if adv and self._adv:
                playlist_noise = self._playlist_noise(pids)
                B_noise = self._B_noise
                #add noise
                playlist_embeds = playlist_embeds + playlist_noise
                B = B + B_noise

            playlist_embeds = my_utils.flatten(playlist_embeds)

            playlist_song_distance = self.execute(playlist_embeds, song_embeds, B, song_biases)
            final_distance = final_distance + playlist_song_distance

            #####################################################

        return -final_distance


class MASR(nn.Module):
    def __init__(self,
                 mdr_model=None,
                 mass_model=None,
                 beta = 0.9):
        super(MASR, self).__init__()

        self._mass = mass_model
        self._mdr = mdr_model
        self._beta = beta


    def _get_l2_loss(self):
        return (self._mdr._get_l2_loss(), self._mass._get_l2_loss())

    def _reset_weights(self):
        pass

    def forward(self,  x, uids, sids, pids=None):

        dist1 = self._mdr(x=None, uids=uids, sids=sids, pids=pids) # distance between target item j and user u
        dist2 = self._mass(x, uids, sids, pids)  # distance between target item j and prev consumed items x

        dist1 = dist1.view(-1, 1)
        dist2 = dist2.view(-1, 1)
        return (1-self._beta)*dist1 + self._beta*dist2











