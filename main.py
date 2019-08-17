import Net as memnet
import pytorch_utils as my_utils
import time
import data_loader as data_loader
import os
from model import REC
import argparse
import multiprocessing
import global_constants as gc

cpus_count = multiprocessing.cpu_count()

parser = argparse.ArgumentParser("Description: Running recommendation baselines")
parser.add_argument('--saved_path', default='chk_points', type=str)
parser.add_argument('--load_best_chkpoint', default=1, type=int, help='loading the best checking point from previous run? (yes/no)')

parser.add_argument('--path', default='data', help='Input data path', type=str)
parser.add_argument('--dataset', default='demo',
                    help='Dataset name', type=str)

parser.add_argument('--epochs', default=50, help='Number of epochs to run', type=int)
parser.add_argument('--batch_size', default=256, help='Batch size', type=int)
parser.add_argument('--num_factors', default=64, help='number of latent factors', type=int)

parser.add_argument('--reg_mdr', nargs='?', default='0.0', help ='Regularization for users and item embeddings', type=str) #update reg to 0.01 for ml100k
parser.add_argument('--reg_mass', nargs='?',  default='0.0', help ='Regularization for users and item embeddings', type=str) #update reg to 0.01 for ml100k
parser.add_argument('--num_neg', default=4, type=int, help='Number of negative instances for each positive sample')
parser.add_argument('--lr', default=0.001, type=float, help = 'Learning rate') #0.001 0.01 works well for amazon dataset
parser.add_argument('--loss_type', nargs='?', default='bpr',
                        help='Specify a loss function: bce, pointwise, bpr, hinge, adaptive_hinge')
parser.add_argument('--distance_metric', default='l2', help='Selecting the distance metric (l2, l1)', type=str)
parser.add_argument('--max_seq_len', type=int, default=-1, help='maximum number of users/items to represents for a item/user')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout probability for dropout layer')
parser.add_argument('--act_func', default='relu', type=str,
                    help='activation function [none, relu, tanh], '
                         'use when combing target user u, target item j and consumed item i')
parser.add_argument('--act_func_mdr', default='none', type=str)
parser.add_argument('--n_layers_mdr', default=1, type=int)

parser.add_argument('--beta', default=0.5, type=float, help='contribution of MDR in the MASR')

parser.add_argument('--topk', type=int, default=10, help='evaluation top K such as: NDCG@K, HITS@K')


parser.add_argument('--model', default='masr', help='Selecting the model type [mass, masr, mdr], masr = mass + mdr', type=str) # masr: deep metric memory recommender

parser.add_argument('--optimizer', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')



# parser.add_argument('--layers', nargs='?', default='[64,32,16,8]', help ='Regularization for users and item embeddings', type=str)

parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')

parser.add_argument('--eval', type=int, default=0,
                        help='Whether to evaluate the saved check points only or to re-build the model')


parser.add_argument('--cuda', type=int, default=0,
                        help='using cuda or not')

parser.add_argument('--seed', type=int, default=98765,
                        help='random seed')


parser.add_argument('--decay_step', type=int, default=50, help='how many steps to decay the learning rate')
parser.add_argument('--decay_weight', type=float, default=0.5, help ='percent of decaying')

parser.add_argument('--data_type', default='upt', type=str, help ='data set type: upt, pt, or ut')
parser.add_argument('--data_type_mdr', default='upt', type=str, help ='data set type: upt, pt, or ut')
parser.add_argument('--data_type_mass', default='ut', type=str, help ='data set type: upt, pt, or ut')
parser.add_argument('--debug', default=0, type=int, help ='debugging mode, data is shortened for observation')

parser.add_argument('--init_transform_type', default = 'identity', type=str, help='init transformation weights Wa,'
                                                                                  'Wb, Wc, Wd. '
                                                                                  'options:identity, '
                                                                                  'he-normal, he-uniform, '
                                                                                  'normal, xavier, lecun')
parser.add_argument('--adv', default=1, type=int, help='Training with adversarial learning or not.')
parser.add_argument('--reg_noise', default=1.0, type=float, help='Noise Regularization.')
parser.add_argument('--eps', default=1.0, type=float, help='Noise magnitude')


args = parser.parse_args()
# args.layers = eval(args.layers)
args.reg_mdr = eval(args.reg_mdr) #can be reg for users, and reg for items, but set to one value for simplification
args.reg_mass = eval(args.reg_mass) #can be reg for users, and reg for items, but set to one value for simplification



#save to global constant
gc.BATCH_SIZE = args.batch_size
gc.DEBUG = bool(args.debug)
gc.SEED = args.seed
gc.model_type = args.model
gc.INIT_TRANSFORM_TYPE = args.init_transform_type

if gc.DEBUG: args.epochs=2
#reuse from neural collaborative filtering
def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user_playlist = arr[0]
            (user, playlist) = user_playlist.split(':')
            user, item, playlist = int(user), int(arr[1]), int(playlist)
            ratingList.append([user, item, playlist])
            line = f.readline()
    return ratingList

#reuse from neural collaborative filtering
def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            negativeList.append(negatives)
            #for j in range(100):
            #    negatives.append(int(iid))
            line = f.readline()
    return negativeList

train_file = os.path.join(args.path, args.dataset, '%s.train.rating'%args.dataset )
vad_file = os.path.join(args.path, args.dataset, '%s.vad.rating'%args.dataset )
vad_neg_file = os.path.join(args.path, args.dataset, '%s.vad.negative'%args.dataset )
test_file = os.path.join(args.path, args.dataset, '%s.test.rating'%args.dataset )
test_neg_file = os.path.join(args.path, args.dataset, '%s.test.negative'%args.dataset )

#vadNegatives, testNegatives = None, None
vadRatings = load_rating_file_as_list(vad_file)
vadNegatives = load_negative_file(vad_neg_file)
testRatings = load_rating_file_as_list(test_file)
testNegatives = load_negative_file(test_neg_file)

print args


rec_model = REC(loss=args.loss_type, #'pointwise, bpr, hinge, adaptive_hinge'
                n_factors = args.num_factors,
                n_iter = args.epochs,
                batch_size = args.batch_size,
                reg_mdr=args.reg_mdr,    # L2 regularization
                reg_mass=args.reg_mass,    # L2 regularization
                lr = args.lr, # learning_rate
                decay_step = args.decay_step, #step to decay the learning rate
                decay_weight = args.decay_weight, #percentage to decay the learning rat.
                optimizer_func = None,
                use_cuda = args.cuda,
                random_state = None,
                num_neg_samples = args.num_neg, #number of negative samples for each positive sample.
                dropout=args.dropout,
                distance_metric = args.distance_metric,
                activation_func = args.act_func,
                activation_func_mdr = args.act_func_mdr,
                n_layers_mdr=args.n_layers_mdr,
                model = args.model,
                beta = args.beta, args= args)


MAX_SEQ_LEN = args.max_seq_len
gc.MAX_SEQ_LEN = MAX_SEQ_LEN

t0 = time.time()
t1 = time.time()
print 'parsing data'
train_iteractions = data_loader.load_data(train_file, dataset=args.dataset)
t2 = time.time()
print 'loading data time: %d (seconds)'%(t2-t1)

print 'building the model'
try:
    rec_model.fit(train_iteractions,
                      verbose=True, topN=10,
                      vadRatings = vadRatings, vadNegatives = vadNegatives,
                      testRatings=testRatings, testNegatives=testNegatives,
                      max_seq_len=MAX_SEQ_LEN, args=args)

except KeyboardInterrupt:
    print 'Exiting from training early'

t_end = time.time()
print 'Total running time: %d (seconds)'%(t_end-t0)
