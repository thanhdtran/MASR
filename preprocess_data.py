import time
import pandas as pd
import os
import numpy as np
import argparse
import global_constants as gc
parser = argparse.ArgumentParser("Description: Running preprocess")
parser.add_argument('--path', default='data', type=str, help='path to the data')
parser.add_argument('--dataset', default='aotm-upt-sampled', type=str, help='ml1m, ml100k, amazon-automotive, amazon-videogames, amazon-toysandgames, amazon-officeproducts')
parser.add_argument('--k_cores', default=5, type=int, help='filter users with at least k items, filter items with at least k users')
parser.add_argument('--preprocess', default=0, type=int, help='Doing preprocessing or not') #0 for tafeng
parser.add_argument('--one_side', default=0, type=int, help='Filter user or both user and item')
parser.add_argument('--start_at', default=1, type = int, help='ID starts at 1 or 0, default is 1, 0 is reserved for padding')
parser.add_argument('--is_test', default=0, type=int, help='make test data, sample some small amount')
args = parser.parse_args()

NUM_NEGS = 100
data_file = os.path.join(args.path, args.dataset, '%s.csv'%args.dataset)

print '\n**********************************'
print 'working on ...', data_file
#data_file = '/media/thanhtd/Andrew/thanh-repo/python_workspace/Deep-Baselines/data/amazon-instantvideo/test.csv'
df = pd.read_csv(data_file, sep=',', header=None)
df.drop_duplicates(subset=[0,1,2],keep=False, inplace=True)
np.random.seed(gc.SEED)
class DataRow:
    def __init__(self, uid, iid, rating, timestamp=None):
        self._uid = uid
        self._iid = iid
        self._rating = rating
        self._timestamp = timestamp


class DataSplitter:
    def __init__(self, df):
        self._df = df

    def split(self):

        df = self._df
        df = df.sort_values(by=[0, 3], ascending=True)  # sort by user_id, then timestamp
        # nrow = df.shape[0]

        trains, vads, tests = [], [], []
        test_neg_strs = []
        vad_neg_strs = []

        user_transactions_map = dict() #for each user and the playlist, store all the items.

        uids_playlists, iids, timestamps, ratings = np.asarray(df[0]), np.asarray(df[1]), \
                                                 np.asarray(df[2]), np.asarray(df[3], dtype=int)
        # uniq_iids = set(iids) # take all the items however, some will appear only one --> better if we take iids from training data only
                            # --> comment out this line
        uniq_iids_train = set()
        train_idxs, test_idxs, vad_idxs = [], [], []

        #step1: store all user's transactions into a map, key is the transaction_id = user + ':' + ts
        for i, uid_playlist in enumerate(uids_playlists):
            iid = iids[i]
            rating = ratings[i]
            ts = timestamps[i]
            key = uid_playlist #uid_playlist is the mixture of user id and (playlist_id or timestamp)
            if key not in user_transactions_map:
                #new transaction:
                transactions = []
                transactions.append((uid_playlist, iid, rating, ts))
                user_transactions_map[key] = transactions
            else:
                transactions = user_transactions_map[key]
                transactions.append((uid_playlist, iid, rating, ts))
                user_transactions_map[key] = transactions

        #key is uid_playlist, transactions contain the list of all transactions of users with format (uid_playlist, iid, rating, ts)
        keys_trans_list = [(key, transactions) for key, transactions in user_transactions_map.items()] #make a list of (key, transactions)
        keys_trans_list = sorted(keys_trans_list)
        for key, transactions in keys_trans_list:
            transactions = np.asarray(transactions)
            trans_len = len(transactions)

            # observed_items = [iid for _,iid, _, _ in transactions]
            # observed_items = set(observed_items)

            if trans_len < 3:
                #adding all as training data:
                train_idxs.append(range(trans_len))
                vad_idxs.append(-1)
                test_idxs.append(-1)
                pass
            else:
                idx = range(trans_len)

                #randomly picking testing example and validation example
                test_vad_idx = np.random.choice(idx, size=2, replace=False)

                test_idx = test_vad_idx[1]
                test_idxs.append(test_idx)

                vad_idx = test_vad_idx[0]
                vad_idxs.append(vad_idx)

                #the rest is for training
                train_idx = [index for index in idx if index not in test_vad_idx]
                train_idxs.append(train_idx)


                #this will be used to make sure if the item is in the training set or not
                uniq_iids_train.update(set([transactions[train_pos][1] for train_pos in train_idx]))




        n_skip_vad = 0
        n_skip_test = 0

        t1 = time.time()
        for i, (key, transactions) in enumerate(keys_trans_list):
            if i%500 == 0:
                t2 = time.time()
                print '%d transactions passed in %d seconds'%(i, t2-t1)
            transactions = np.asarray(transactions)
            trans_len = len(transactions)

            observed_items = [iid for _,iid, _, _ in transactions]
            observed_items = set(observed_items)

            # if i >= 20:
            #     break

            if trans_len < 3:
                #adding all as training data:
                trains.append(transactions)
            else:
                test_idx = test_idxs[i]
                #tests.append(transactions[test_idx])

                # vad_idx = np.random.choice(idx, size=1, replace=False)
                vad_idx = vad_idxs[i]
                #vads.append(transactions[vad_idx])

                #the rest is for training
                train_idx = train_idxs[i]
                trains.append(transactions[train_idx])

                #making neg_items for each interaction for dev and test datasets:
                neg_items = uniq_iids_train - observed_items


                # uid_playlist = transactions[0][0]
                uid_playlist = key


                vad_iid = transactions[vad_idx][1]
                if vad_iid in uniq_iids_train:
                    vads.append(transactions[vad_idx])
                    #only consider the iid in the validation/development that existed in training item ids
                    vad_neg_strs.append(self.make_neg_str(neg_items, uid_playlist, vad_iid,  NUM_NEGS))
                else:
                    n_skip_vad += 1 #just for counting how many samples we skip

                test_iid = transactions[test_idx][1]
                if test_iid in uniq_iids_train:
                    tests.append(transactions[test_idx])
                    # only consider the iid in the validation/development that existed in training item ids
                    test_neg_strs.append(self.make_neg_str(neg_items, uid_playlist, test_iid, NUM_NEGS))
                else:
                    n_skip_test += 1 #just for counting how many samples we skip

        print 'Number of skipped samples in development dataset:', n_skip_vad
        print 'Number of skipped samples in test dataset:', n_skip_test

        unzip_trains = []
        for i in range(len(trains)):
            for train_sample in trains[i]:
                unzip_trains.append(train_sample)


        return unzip_trains, vads, tests, vad_neg_strs, test_neg_strs

    def split_old(self):
        df = self._df
        df = df.sort_values(by=[0, 3], ascending=True) #sort by user_id, then timestamp
        nrow = df.shape[0]
        uids, iids, ratings, timestamps = np.asarray(df[0]), np.asarray(df[1]), np.asarray(df[2]), np.asarray(df[3], dtype=int)
        uniq_uids, index = np.unique(uids, return_index=True)
        uniq_iids = set(iids)
        num_neg = NUM_NEGS

        test_idx = []
        train_idx = []
        vad_idx = []
        test_neg_strs = []
        vad_neg_strs = []

        #print index
        for i in range(len(index)):


            if i < (len(index)-1):
                test_idx_tmp = index[i+1]-1 #reseve last lement for testing data
            else: #last element in index:
                test_idx_tmp = nrow-1
            if (test_idx_tmp + 1 - index[i]) <= 2: #args.k_cores:
                #a user has less than k_cores items --> do something: take all as training
                for idx in range(index[i], test_idx_tmp+1):
                    train_idx.append(idx)
                continue

            test_idx.append(test_idx_tmp)

            #print 'test_idx:',test_idx_tmp
            train_vad_idx_fro, train_vad_idx_to = index[i], test_idx_tmp-1
            #print train_vad_idx_fro, train_vad_idx_to
            vad_idx_tmp = np.random.random_integers(low=train_vad_idx_fro, high= train_vad_idx_to, size=1)[0]
            vad_idx.append(vad_idx_tmp)
            #print 'vad_idx:',vad_idx_tmp

            train_idx_tmp = []
            for idx in range(train_vad_idx_fro, test_idx_tmp):
                ##print 'test,',idx, vad_idx_tmp
                if idx != vad_idx_tmp:
                    train_idx.append(idx)
                    train_idx_tmp.append(idx)
            #print 'train idx:',train_idx_tmp

            #sample negative items for users:
            neg_items = uniq_iids - set(iids[train_vad_idx_fro:(test_idx_tmp+1)])
            #print set(iids[train_vad_idx_fro:(test_idx_tmp+1)])
            #print neg_items
            #print index[i],nrow
            #test_user_item_str = '(' + str(uids[index[i]]) + ',' + str(iids[test_idx_tmp])  + ')'
            #rand_neg_items = np.random.choice(list(neg_items), size=num_neg, replace=False)
            #for neg_item in rand_neg_items:
            #    test_user_item_str += '\t' + str(neg_item)
            #test_user_item_str += '\n'
            #test_neg_strs.append(test_user_item_str)
            #print test_user_item_str
            test_neg_strs.append(self.make_neg_str(neg_items, uids[index[i]], iids[test_idx_tmp], num_neg))
            vad_neg_strs.append(self.make_neg_str(neg_items, uids[index[i]], iids[vad_idx_tmp], num_neg))



        test_df = df.iloc[test_idx]
        train_df = df.iloc[train_idx]
        vad_df = df.iloc[vad_idx]

        return train_df, vad_df, test_df, test_neg_strs, vad_neg_strs

    def make_neg_str(self, neg_items, uid, iid, num_neg=99):
        test_neg_str = '(' + str(uid) + ',' + str(iid) + ')'
        rand_neg_items = np.random.choice(list(neg_items), size=num_neg, replace=False)
        for neg_item in rand_neg_items:
            test_neg_str += '\t' + str(neg_item)
        test_neg_str += '\n'
        return test_neg_str


    def make_rows(self, uids, iids, ratings, timestamps):
        l = len(uids)
        rows = []
        for i in range(l):
            row = self.make_row(uids[i], iids[i], ratings[i], timestamps[i])
            rows.append(row)
        return rows

    def make_row(self, uid, iid, rating, timestamp):
        return DataRow(uid, iid, rating, timestamp)


def numerize(df, uid_map, playlistid_map1, iid_map):
    uid = []
    max_uid = 0
    for uid_playlist in np.asarray(df[0]):
        old_uid = int(uid_playlist.split(':')[0])
        playlist_id = int(uid_playlist.split(':')[1])
        new_uid = uid_map[old_uid]
        max_uid = max(new_uid, max_uid)
        new_playlistid = playlistid_map1[playlist_id]
        uid.append(str(new_uid) + ':' + str(new_playlistid))
    # uid = map(lambda x: str(uid_map[x.split(':')[0]]) + ':' + str(x.split(':')[1]) , df[0]) #keep the transaction id
    uid = np.asarray(uid)
    sid = map(lambda x: iid_map[x], df[1])
    df[0] = uid
    df[1] = sid
    return df
def get_uniq_uids(df):
    uniq_user_playlists = df[0].unique()
    uids = set()
    transaction_ids = set()
    for user_playlist in uniq_user_playlists:
        uids.add(int(user_playlist.split(':')[0]))
        transaction_ids.add(int(user_playlist.split(':')[1]))
    return np.asarray(sorted(list(uids))), np.asarray(sorted(list(transaction_ids)))

def get_count(df, id):
    count_groupbyid = df[[id]].groupby(id, as_index=False)
    count = count_groupbyid.size()
    # print count.index[count >= 2]
    return count
#remove users who had less than min_pc interactions, and items with less than min_uc users:
def filter(df, min_pc=5, min_uc=5):
    #keep users who backed at least min_pc projects
    print 'filtering condition:', min_pc, min_uc
    current_size = 0
    next_size = df.shape[0]
    iter = 1

    uid_local_map = dict(((uid, i + START_AT) for i, uid in enumerate(df[0].unique())))
    uid_local_rev_map = dict(((i + START_AT, uid) for i, uid in enumerate(df[0].unique())))
    uids = map(lambda x: uid_local_map[x], df[0])
    df[0] = uids

    while(current_size != next_size):
        print 'filter with loop %d, size: %d, num_users: %d, num_items: %d'%(iter, df.shape[0], len(df[0].unique()), len(df[1].unique()) )

        projectcount = get_count(df, 1)  # 1 is for itemId

        stop_cond = 0

        #if args.is_test:
        #    if len(df[0].unique())< 1000 and len(df[1].unique()) < 1000: break
        print 'each item is consumed by at least: ',np.min(projectcount.values)
        if np.min(projectcount.values) < min_uc and min_uc > 0:
           print 'filtering items ...'
           df = df[df[1].isin(projectcount.index[projectcount >= min_uc])]
        else:
            stop_cond += 1

        usercount = get_count(df, 0)  # 0 is for user_id
        print 'each user consumed at least:',np.min(usercount.values)


        if not args.one_side:
            print 'filtering users ...'
            if np.min(usercount.values) <  min_pc and min_pc > 0:
                df = df[df[0].isin(usercount.index[usercount >= min_pc])]
            else:
                stop_cond += 1

            if stop_cond >= 2: break
        break
        if stop_cond >= 1: break
#        if iter >= 2: break # early stop, one-time filtering?
    usercount, projectcount = get_count(df, 0), get_count(df, 1)
    print 'After filtering #interactions/user: %.2f, min #interactions/user: %d'%(np.mean(usercount.values), np.min(projectcount.values))
    print 'After filtering #interactions/item: %.2f, min #interactions/item: %d'%(np.mean(projectcount.values), np.min(projectcount.values))

    uids_rev = map(lambda x: uid_local_rev_map[x], df[0])
    df[0] = uids_rev
    return df, usercount, projectcount

t1 = time.time()
(uniq_uids, uniq_transactions), uniq_iids = get_uniq_uids(df), df[1].unique()
print 'Before filtering: total users: %d, total items: %d'%(len(uniq_uids), len(uniq_iids))
usercount, projectcount = get_count(df, 0), get_count(df, 1)
print 'Before filtering: #interactions/user: %.2f, min #interactions/user: %d'%(np.mean(usercount.values), np.min(usercount.values))
print 'Before filtering: #interactions/item: %.2f, min #interactions/item: %d'%(np.mean(projectcount.values), np.min(projectcount.values))

# if args.dataset != 'ml1m0' : # and args.dataset != 'ml100k'\
#     print 'do filtering with k-cores'
(uniq_uids, uniq_transactions), uniq_iids  = get_uniq_uids(df), sorted(df[1].unique())
START_AT = int(args.start_at)
uid_map1 = dict(((uid, i+START_AT) for i, uid in enumerate(uniq_uids))) #aware of the transaction_id
iid_map1 = dict(((iid, i+START_AT) for i, iid in enumerate(uniq_iids)))
playlistid_map1 = dict(((playlist_id, i+START_AT) for i, playlist_id in enumerate(uniq_transactions)))

df = numerize(df, uid_map1, playlistid_map1, iid_map1)

if args.preprocess:
    df, _, _ = filter(df, min_pc=args.k_cores, min_uc=args.k_cores)
# print df

(uniq_uids, uniq_transactions), uniq_iids  = get_uniq_uids(df), sorted(df[1].unique())
uid_map2 = dict(((uid, i+START_AT) for i, uid in enumerate(uniq_uids)))
iid_map2 = dict(((iid, i+START_AT) for i, iid in enumerate(uniq_iids)))
playlistid_map2 = dict(((playlist_id, i+START_AT) for i, playlist_id in enumerate(uniq_transactions)))
df = numerize(df, uid_map2, playlistid_map2, iid_map2)

df_tmp = df
df_tmp = df.sort_values(by=[0,3], ascending=True)
path = os.path.join(args.path, args.dataset)
saved_all_file = os.path.join(path, '%s.ratings'%args.dataset)
df_tmp.to_csv(saved_all_file, sep='\t', header=None, index=False)

def write_mapping(map, file_out):
    with open(file_out, 'w') as f:
        sep = '\t'
        for (k, v) in map.items():
            f.write(str(k) + sep + str(v) + '\n')
        f.flush()
        f.close()
def remapping_and_save(uid_map1, iid_map1, uid_map2, iid_map2):
    # purpose: save for further visualization

    #reverse mapping from map1: from real_id --> encoded_id, we need encoded_id --> real_id
    reversed_uid_map1 = dict((e1, r) for (r, e1) in uid_map1.items())
    reversed_iid_map1 = dict((e1, r) for (r, e1) in iid_map1.items())
    uid_map = dict((reversed_uid_map1[e1], e2) for (e1, e2) in uid_map2.items()) # r --> e2
    iid_map = dict((reversed_iid_map1[e1], e2) for (e1, e2) in iid_map2.items()) # r --> e2

    reversed_uid_map = dict((e2, reversed_uid_map1[e1]) for (e1, e2) in uid_map2.items()) # e2 --> r
    reversed_iid_map = dict((e2, reversed_iid_map1[e1]) for (e1, e2) in iid_map2.items()) # e2 --> r
    #save to file:
    uid_mapping_file = os.path.join(args.path, args.dataset, '%s_encoded_userids.csv'%args.dataset)
    re_uid_mapping_file = os.path.join(args.path, args.dataset, '%s_decoded_userids.csv'%args.dataset)

    iid_mapping_file = os.path.join(args.path, args.dataset, '%s_encoded_itemids.csv'%args.dataset)
    re_iid_mapping_file = os.path.join(args.path, args.dataset, '%s_decoded_itemids.csv'%args.dataset)

    write_mapping(uid_map, uid_mapping_file)
    write_mapping(reversed_uid_map, re_uid_mapping_file)

    write_mapping(iid_map, iid_mapping_file)
    write_mapping(reversed_iid_map, re_iid_mapping_file)

# print 'saving mappings ...'
# remapping_and_save(uid_map1, iid_map1, uid_map2, iid_map2)

print 'After filtering: total users: %d, total items: %d'%(len(uniq_uids), len(uniq_iids))
#sort data frame by timestamp

df = df.sort_values(by=[0,3], ascending=True) #sort by user then by timestamp
#print df
print 'total interactions:',df.shape[0]
#
#print np.unique(np.asarray(df[0]), return_index=True )
print 'splitting training, testing, validation ...'
data_spliter = DataSplitter(df)
trains, vads, tests, vad_neg_strs, test_neg_strs = data_spliter.split()
#print test_neg_strs
#print vad_neg_strs

#flush to file:

path = os.path.join(args.path, args.dataset)
train_file = os.path.join(path, '%s.train.rating'%args.dataset)
vad_file = os.path.join(path, '%s.vad.rating'%args.dataset)
vad_neg_file = os.path.join(path, '%s.vad.negative'%args.dataset)
test_file = os.path.join(path, '%s.test.rating'%args.dataset)
test_neg_file = os.path.join(path, '%s.test.negative'%args.dataset)

with open(train_file, 'w') as f:
    i = 0
    for (uid_playlist, iid, rating, ts) in trains:

        i += 1
        line = str(uid_playlist) + '\t' + str(iid) + '\t' + str(rating) + '\t' + str(ts) + '\n'
        f.write(line)
        if i%1000 == 0: f.flush()
    f.flush()
f.close()

with open(vad_file, 'w') as f:
    i = 0
    for (uid_playlist, iid, rating, ts) in vads:

        i += 1
        line = str(uid_playlist) + '\t' + str(iid) + '\t' + str(rating) + '\t' + str(ts) + '\n'
        f.write(line)
        if i%1000 == 0: f.flush()
    f.flush()
f.close()

with open(test_file, 'w') as f:
    i = 0
    for (uid_playlist, iid, rating, ts) in tests:
        i += 1
        line = str(uid_playlist) + '\t' + str(iid) + '\t' + str(rating) + '\t' + str(ts) + '\n'
        f.write(line)
        if i%1000 == 0: f.flush()
    f.flush()
f.close()

with open(vad_neg_file,'w') as f:
    for vad_neg_str in vad_neg_strs:
        f.write(vad_neg_str)
    f.flush()
with open(test_neg_file,'w') as f:
    for test_neg_str in test_neg_strs:
        f.write(test_neg_str)
    f.flush()

t2 = time.time()
print 'Total time: %d secs'%(t2-t1)

