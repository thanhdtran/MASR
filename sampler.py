"""
Module containing functions for negative item sampling.
"""

import numpy as np
from scipy.sparse import csr_matrix
import global_constants as gc
np.random.seed(gc.SEED)

class Sampler(object):
    def __init__(self):
        super(Sampler, self).__init__()
        self.user_neg_items_map = dict()
        self.n_users = 0
        self.n_items = 0
        self.user_items_map = None
        self.playlist_items_map = None

    def init_user_item_seqs(self, user_all_items, num_users, num_items):
        self.n_users = num_users
        self.n_items = num_items
        self.user_items_map = user_all_items


    def random_neg_items(self, user_ids=None, num_neg=4):
        neg_items = np.zeros(shape=(len(user_ids), num_neg), dtype=np.int64)
        for i, uid in enumerate(user_ids):
            user_pos_items = self.user_items_map[uid]
            local_neg_items = set()
            j = 0
            while j < num_neg:
                neg_item = np.random.randint(self.n_items)
                if neg_item not in user_pos_items and neg_item not in local_neg_items and neg_item != gc.PADDING_IDX:
                    local_neg_items.add(neg_item)
                    neg_items[i][j] = neg_item
                    j += 1
        return neg_items

    def random_sample_items(self, num_items, shape, user_ids = None, random_state=None):
        """
        Randomly sample a number of items. We assume that a song is added into one playlist only
        or an item is added into one basket only.

        Parameters
        ----------

        num_items: int
            Total number of items from which we should sample:
            the maximum value of a sampled item id will be smaller
            than this.
        shape: int or tuple of ints
            Shape of the sampled array.
        random_state: np.random.RandomState instance, optional
            Random state to use for sampling.
            shape: (number of users, number of items)

        Returns
        -------

        items: np.array of shape [shape]
            Sampled item ids.
        """
        if user_ids is not None:
            items = np.zeros(shape, dtype=np.int64)
            num_neg_item_per_user = 1
            if isinstance(shape, tuple):
                num_neg_item_per_user = shape[-1]
            # else:
            for i, uid in enumerate(user_ids):
                # neg_items = set(range(num_items)) - set(self.user_seqs_nopad[uid])
                items[i] = np.random.choice(self.user_neg_items_map[uid], num_neg_item_per_user,replace=True)
            return items
        else:
            if random_state is None:
                random_state = np.random.RandomState()

            items = random_state.randint(1, num_items, shape, dtype=np.int64) #random from 1 to num_items as 0 is PADDING_IDX

            return items
