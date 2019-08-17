'''
A general data loader, to load different kinds of data, as long as the format is:
user_id \t item_id \t rating.
'''
import numpy as np
import pandas as pd
import scipy.sparse as sp
import global_constants as gc
from collections import defaultdict

np.random.seed(gc.SEED)
def _sliding_window(tensor, window_size, step_size=1):

    for i in range(len(tensor), 0, -step_size):
        yield tensor[max(i - window_size, 0):i]


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length,
                        step_size):

    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length,
                                   step_size):

            yield (user_ids[i], seq)

def load_data(path, sep = '\t', header=None, dataset=None):
    data = pd.read_csv(path, sep=sep, header=header)
    user_ids = np.asarray(data[0])
    item_ids = np.asarray(data[1])
    ratings = None
    # ratings = data[2] #we don't need it because of implicit feedback dataset.
    timestamps = None
    if data.shape[1] >= 4:
        #contain timestamp
        timestamps = data[3]
    return Interactions(path, dataset=dataset)


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.

    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.

    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.

    Parameters
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.

    Attributes
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps

    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(self,
                 data_path,
                 dataset='ml1m'
                 ):
        self._user_playlist_all_items = defaultdict(list) # get consumed items, key is user:playlist, value is the consumed items
        self._user_all_items = defaultdict(list)  # get consumed items, key is user, value is the consumed items
        self._useritem_prev_items = defaultdict(list) #key is (user,item), value is previous items
        self._user_ids = []
        self._user_playlists = []
        self._playlists = []
        self._item_ids = []
        self._ratings = []
        self._timestamps = []
        self._num_users, self._num_items, self._num_playlists = None, None, None
        self._playlist_all_items = defaultdict(list) #get all items in the same playlist.
        duplicate_user_item_pairs = set() #remove duplicate pairs
        with open(data_path, 'r') as f:
            #user ids and item ids must start at 1
            for line in f:
                tokens = line.strip().split('\t')
                uid_playlist = tokens[0]
                uid = int(uid_playlist.split(':')[0])
                playlist_id = int(uid_playlist.split(':')[1])
                iid, rating, timestamp = int(tokens[1]), float(tokens[3]), int(float(tokens[2]))
                duplicate_key = (uid_playlist, iid)
                if duplicate_key not in duplicate_user_item_pairs: duplicate_user_item_pairs.add((uid_playlist, iid))
                else: continue

                self._user_playlists.append(uid_playlist)
                self._playlists.append(playlist_id)
                self._user_ids.append(uid)
                self._item_ids.append(iid)
                self._ratings.append(rating)
                self._timestamps.append(timestamp)


                prev_items = self._user_playlist_all_items[uid_playlist][:] \
                                if len(self._user_playlist_all_items[uid_playlist]) > 0 else [gc.PADDING_IDX]
                self._useritem_prev_items[(uid_playlist, iid)].extend(prev_items) #adding all previous items in same transactions
                self._user_playlist_all_items[uid_playlist].append(iid)
                self._playlist_all_items[playlist_id].append(iid)
                self._user_all_items[uid].append(iid)


        self._user_ids, self._item_ids = np.asarray(self._user_ids), np.asarray(self._item_ids)
        self._ratings, self._playlists = np.asarray(self._ratings), np.asarray(self._playlists)

        self.num_users = self._num_users or int(np.max(self._user_ids) + 1)
        self.num_items = self._num_items or int(np.max(self._item_ids) + 1)
        self.num_playlists = self._num_playlists or int(np.max(self._playlists) + 1)

        self._max_len_user_seq = 0 #maximum number of consumed items in all users' transactions
        for uid_playlist in set(self._user_playlists):
            self._max_len_user_seq = max(self._max_len_user_seq, len(self._user_playlist_all_items[uid_playlist]) - 1)


        self._dataset = dataset




    def __repr__(self):

        return ('<Interactions dataset ({num_users} users x {num_items} items '
                'x {num_interactions} interactions)>'
                .format(
                    num_users=self.num_users,
                    num_items=self.num_items,
                    num_interactions=len(self)
                ))

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()



    def get_batch_seqs(self, user_ids, item_ids, playlists, max_seq_len=100, type='all'):
        '''

        :param user_ids:
        :param max_seq_len:
        :param type: two options: only_prev, all. all: extract all consumed items,
                                                  only_prev: only items consumed before this target item
        :return:
        '''

        batch_size = len(user_ids)
        seq_len = max_seq_len if max_seq_len != -1 else self._max_len_user_seq
        user_seqs = np.zeros((batch_size, seq_len), dtype=np.int64)
        for i, (uid, iid, playlist) in enumerate(zip(user_ids, item_ids, playlists)):
            user_seq = np.zeros(seq_len, dtype=np.int64)
            if type == 'only_prev':
                key = (str(uid) + ':' + str(playlist), iid)
                tmp_seq = self._useritem_prev_items[key]
            if type == 'all' or len(tmp_seq) == 0:
                #type = all or extracting the seq for rating in testing and validation --> take all consumed items in train
                key = str(uid) + ':' + str(playlist) #to represent for a session
                tmp_seq = np.asarray(self._user_playlist_all_items[key], dtype=np.int64)
                tmp_seq = tmp_seq[tmp_seq != iid]  # remove item iid in user seq

            # shorten the seq as of seq_len limitation
            if len(tmp_seq) > seq_len:
                tmp_seq = tmp_seq[-seq_len:]

            if len(tmp_seq) == 0:
                print 'data_loader.py, line 215: error -->',uid, iid, playlist, seq_len

            user_seq[-len(tmp_seq):] = tmp_seq
            user_seqs[i] = user_seq
        return user_seqs
