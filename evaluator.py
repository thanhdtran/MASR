import sys
import math
import numpy as np

#leave one out evaluation
def evaluate(model, testRatings, testNegatives, K):


    hits, ndcgs = [], []
    i = 0
    for idx in xrange(len(testRatings)):
        i+=1
        (hr, ndcg) = eval_one_rating(model, testRatings, testNegatives, K, idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return np.nanmean(hits), np.nanmean(ndcgs)

def eval_one_rating(model, testRatings, testNegatives, K, idx):
    '''begin_at_0: item_id and user_id begins at 0 or not. Since 0 is padding --> need to add 1:'''
    max_user_id = model._n_users
    max_item_id = model._n_items

    rating = testRatings[idx]
    items = testNegatives[idx][:]
    uid, iid, oid = rating[0], rating[1], rating[2]
    items.append(iid)


    if uid >= max_user_id or iid >= max_item_id:
        # didn't observe in the training data for this user or this item
        # if the item id is all the last item in the transaction --> not observe in the training data.
        return np.nan, np.nan

    items = np.asarray(items)
    items = np.delete(items, np.where(items >= max_item_id)) #remove negative items that didn't appear in the training data if possible

    items = np.asarray(items)
    users = np.full(len(items), uid, dtype=np.int64)
    oids = np.full(len(items), oid, dtype=np.int64)
    if np.sum(items >= max_item_id) >= 1:
        print items
        print 'error because the exiting item id that is not observed in training data.'
        sys.exit(1)

    # print len(items)
    predictions = model.predict(users, items, oids)

    indices = np.argsort(-predictions)[:K] #indices of items with highest scores
    ranklist = items[indices]

    hr = getHitRatio(ranklist, iid)
    ndcg = getNDCG(ranklist, iid)

    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
