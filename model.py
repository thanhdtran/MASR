import global_constants as gc
import sampler as my_sampler
import time
import pytorch_utils as my_utils
import numpy as np
import torch.optim as optim
import losses as my_losses
import torch
import evaluator as my_evaluator
from torch.autograd import Variable
from model_based import ModelBased
from torch.optim.lr_scheduler import StepLR
from Net import MDR, MASS, MASR



class REC(ModelBased):
    def __init__(self, loss='pointwise',
                 n_factors = 8,
                 n_iter = 20,
                 batch_size = 256,
                 reg_mdr= 0.00001,  # L2, L1 regularization
                 reg_mass = 0.0001,    # L2, L1 regularization
                 lr = 1e-2, # learning_rate
                 decay_step = 20,
                 decay_weight= 0.5,
                 optimizer_func = None,
                 use_cuda = False,
                 random_state = None,
                 num_neg_samples = 1, #number of negative samples for each positive sample.
                 dropout=0.5,
                 distance_metric='l2',
                 activation_func = 'none', #relu, or tanh,
                 activation_func_mdr = 'none',
                 n_layers_mdr=1,
                 model='mass',
                 beta=0.5, args=None
                 ):
        super(REC, self).__init__()
        self._args = args
        self._loss = loss
        self._n_factors = n_factors
        self._embedding_size = n_factors

        self._n_iters = n_iter
        self._batch_size = batch_size
        self._lr = lr
        self._decay_step = decay_step
        self._decay_weight = decay_weight

        self._reg_mdr = reg_mdr
        self._reg_mass = reg_mass
        self._optimizer_func = optimizer_func

        self._use_cuda = use_cuda
        self._random_state = random_state or np.random.RandomState()
        self._num_neg_samples = num_neg_samples

        self._n_users = None
        self._n_items = None
        self._lr_schedule = None
        self._loss_func = None
        self._dropout = dropout
        self._distance_metric = distance_metric
        self._gate_tying = gate_tying
        self._model = model
        self._beta = beta


        #my_utils.set_seed(self._random_state.randint(-10**8, 10**8), cuda=self._use_cuda)
        my_utils.set_seed(gc.SEED)

        self._activation_func = activation_func
        self._activation_func_mdr = activation_func_mdr
        self._n_layers_mdr = n_layers_mdr
        #for evaluation during training
        self._sampler = my_sampler.Sampler()

    def _has_params(self):

        for params in self._net.parameters():
            if params.requires_grad:
                return True
        if self._model == 'masr':
            for params in self._net._mdr.parameters():
                if params.requires_grad:
                    return True
            for params in self._net._memnet.parameters():
                if params.requires_grad:
                    return True
        return False

    def _is_initialized(self):
        return self._net is not None

    def _initialize(self, interactions, max_seq_len=-1):
        self._interactions = interactions
        self._max_user_seq_len = interactions._max_len_user_seq if max_seq_len == -1 else max_seq_len
        # self._max_item_seq_len = interactions._max_len_item_seq if max_seq_len == -1 else max_seq_len

        (self._n_users, self._n_items, self._n_playlists) = (interactions.num_users, interactions.num_items, interactions.num_playlists)
        print 'total users: %d, total playlists: %d, total items: %d, total training interactions: %d'%(self._n_users,
                                                                                            self._n_playlists,
                                                                                            self._n_items,
                                                                                            len(interactions._user_ids))

        if self._model == 'mass':
            if self._args.data_type == 'upt':
                self._net = MASS(n_users=self._n_users, n_items=self._n_items, n_playlists = self._n_playlists,
                                          embedding_size=self._embedding_size,
                                          item_seq_size=self._max_user_seq_len, distance_type=self._distance_metric,
                                          nonlinear_func=self._activation_func,
                                          dropout_prob=self._dropout,
                                          adv=self._args.adv, eps=self._args.eps
                                          )
            elif self._args.data_type == 'pt':
                self._net = MASS(n_users=self._n_playlists, n_items=self._n_items, n_playlists=None,
                                          embedding_size=self._embedding_size,
                                          item_seq_size=self._max_user_seq_len,
                                          distance_type=self._distance_metric,
                                          nonlinear_func=self._activation_func,
                                          dropout_prob=self._dropout,
                                          adv=self._args.adv, eps=self._args.eps
                                          )
            else:
                self._net = MASS(n_users=self._n_users, n_items=self._n_items, n_playlists=None,
                                          embedding_size=self._embedding_size,
                                          item_seq_size=self._max_user_seq_len,
                                          distance_type=self._distance_metric,
                                          nonlinear_func=self._activation_func,
                                          dropout_prob=self._dropout,
                                          adv=self._args.adv, eps=self._args.eps
                                          )

        elif self._model == 'mdr':
            if self._args.data_type == 'upt':
                self._net = MDR(n_users=self._n_users, n_items=self._n_items, n_playlists = self._n_playlists,
                                 embedding_size=self._embedding_size,
                                 nonlinear_func=self._activation_func_mdr,
                                 num_layers=self._n_layers_mdr,
                                 dropout_prob=self._dropout,
                                 adv=self._args.adv, eps=self._args.eps)
            elif self._args.data_type == 'pt':
                self._net = MDR(n_users=self._n_playlists, n_items=self._n_items, n_playlists=None,
                                 embedding_size=self._embedding_size,
                                 nonlinear_func=self._activation_func_mdr,
                                 num_layers=self._n_layers_mdr,
                                 dropout_prob=self._dropout,
                                 adv=self._args.adv, eps=self._args.eps)
            else:
                self._net = MDR(n_users=self._n_users, n_items=self._n_items, n_playlists=None,
                                 embedding_size=self._embedding_size,
                                 nonlinear_func=self._activation_func_mdr,
                                 num_layers=self._n_layers_mdr,
                                 dropout_prob=self._dropout,
                                 adv=self._args.adv, eps=self._args.eps)

        else:
            mdr_model = None
            mass_model = None
            if self._args.data_type_mdr == 'upt':
                mdr_model = MDR(n_users=self._n_users, n_items=self._n_items, n_playlists=self._n_playlists,
                                 embedding_size=self._embedding_size,
                                 nonlinear_func=self._activation_func_mdr,
                                 num_layers=self._n_layers_mdr,
                                 dropout_prob=self._dropout)
            elif self._args.data_type_mdr == 'pt':
                mdr_model = MDR(n_users=self._n_playlists, n_items=self._n_items, n_playlists=None,
                                 embedding_size=self._embedding_size,
                                 nonlinear_func=self._activation_func_mdr,
                                 num_layers=self._n_layers_mdr,
                                 dropout_prob=self._dropout)
            else:
                mdr_model = MDR(n_users=self._n_users, n_items=self._n_items, n_playlists=None,
                                 embedding_size=self._embedding_size,
                                 nonlinear_func=self._activation_func_mdr,
                                 num_layers=self._n_layers_mdr,
                                 dropout_prob=self._dropout)

            if self._args.data_type_mass == 'upt':
                mass_model = MASS(n_users=self._n_users, n_items=self._n_items, n_playlists = self._n_playlists,
                                          embedding_size=self._embedding_size,
                                          item_seq_size=self._max_user_seq_len, distance_type=self._distance_metric,
                                          nonlinear_func=self._activation_func,
                                          dropout_prob=self._dropout
                                          )
            elif self._args.data_type_mass == 'pt':
                mass_model = MASS(n_users=self._n_playlists, n_items=self._n_items, n_playlists=None,
                                          embedding_size=self._embedding_size,
                                          item_seq_size=self._max_user_seq_len,
                                          distance_type=self._distance_metric,
                                          nonlinear_func=self._activation_func,
                                          dropout_prob=self._dropout
                                          )
            else:
                mass_model = MASS(n_users=self._n_users, n_items=self._n_items, n_playlists=None,
                                          embedding_size=self._embedding_size,
                                          item_seq_size=self._max_user_seq_len,
                                          distance_type=self._distance_metric,
                                          nonlinear_func=self._activation_func,
                                          dropout_prob=self._dropout
                                          )

            self._net = MASR(mdr_model=mdr_model, mass_model=mass_model, beta=self._args.beta)

        self._net = my_utils.gpu(self._net, self._use_cuda)

        reg = 1e-6
        if self._args.model == 'mdr': reg = self._args.reg_mdr
        elif self._args.model == 'mass': reg = self._args.reg_mass
        else: reg = 1e-6
        print 'setting reg to :', reg
        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=reg,
                lr=self._lr
            )
            decay_step = self._decay_step
            decay_percent = self._decay_weight
            self._lr_schedule = StepLR(self._optimizer, step_size=decay_step, gamma=decay_percent)

        else:
            self._optimizer = self._optimizer_func(self._net.parameters())
        if self._loss == 'pointwise':
            self._loss_func = my_losses.pointwise_loss
        elif self._loss == 'bpr':
            self._loss_func = my_losses.bpr_loss
        elif self._loss == 'hinge':
            self._loss_func = my_losses.hinge_loss
        elif self._loss == 'bce': #binary cross entropy
            self._loss_func = my_losses.pointwise_bceloss
        else:
            self._loss_func = my_losses.adaptive_hinge_loss

        #self._sampler.init_user_item_seqs(interactions._user_all_items, interactions.num_users, interactions.num_items)
        self._sampler.init_user_item_seqs(interactions._playlist_all_items, interactions.num_playlists, interactions.num_items)



    def fit(self, interactions, verbose=False, topN=10,
            vadRatings=None, vadNegatives=None,
            testRatings=None, testNegatives=None,
            max_seq_len=100, args=None):

        if not self._is_initialized():
            self._initialize(interactions, max_seq_len=max_seq_len)

        user_ids = interactions._user_ids.astype(np.int64)
        item_ids = interactions._item_ids.astype(np.int64)
        playlist_ids = interactions._playlists.astype(np.int64)

        # self._check_input(sequences)
        if verbose:
            best_hit = 0.0
            best_ndcg = 0.0
            best_epoch = 0
            test_hit, test_ndcg = 0.0, 0.0

        if args.load_best_chkpoint > 0:
            # print self._net.parameters
            best_hit, best_ndcg = self.load_checkpoint(args)
            print 'Results from best checkpoints ...'
            # print self._net.parameters
            if verbose:
                t1 = time.time()
                hits, ndcg = my_evaluator.evaluate(self, vadRatings, vadNegatives, topN)
                for topN in range(10, 11, 1):
                    hits_test, ndcg_test = my_evaluator.evaluate(self, testRatings, testNegatives, topN)
                    t2 = time.time()
                    eval_time = t2 - t1
                    print('| Eval time: %d '
                          '| Vad hits@%d = %.3f | Vad ndcg@%d = %.3f '
                          '| Test hits@%d = %.3f | Test ndcg@%d = %.3f |'
                          % ( eval_time, topN, hits, topN, ndcg, topN, hits_test, topN,
                             ndcg_test))
                    if topN == 10:
                        test_hit = hits_test
                        test_ndcg = ndcg_test
                        best_hit, best_ndcg = hits, ndcg
                topN = 10
            print 'End!'

        if args.eval:
            print 'Evaluation using the saved checkpoint done!'
            return

        if self._has_params():
            for epoch in range(self._n_iters):

                self._lr_schedule.step(epoch)

                self._net.train()  # set training environment
                t1 = time.time()
                users, items, playlists = my_utils.shuffle(user_ids,
                                                item_ids,
                                                playlist_ids,
                                                random_state=self._random_state)
                users = np.asarray(users)
                items = np.asarray(items)
                playlists = np.asarray(playlists)
                #neg_items = self._sampler.random_neg_items(users, num_neg=args.num_neg)
                neg_items = self._sampler.random_neg_items(playlists, num_neg=args.num_neg)

                epoch_loss = 0.0
                epoch_noise_loss = 0.0

                t1 = time.time()
                total_interactions = 0
                for (minibatch_idx, (batch_user, batch_item, batch_playlists, batch_neg_items)) in enumerate(my_utils.minibatch(
                                                                                                users,
                                                                                                items,
                                                                                                playlists,
                                                                                                neg_items,
                                                                                                batch_size=self._batch_size)):
                    total_interactions += len(batch_user)  # or batch_size

                    if args.model == 'mass' or args.model == 'masr':
                        item_seqs = interactions.get_batch_seqs(batch_user, batch_item, batch_playlists,
                                                                max_seq_len=self._max_user_seq_len, type='all')
                        item_seqs_var = Variable(my_utils.gpu(my_utils.numpy2tensor(item_seqs).type(torch.LongTensor),
                                                              self._use_cuda), requires_grad=False)
                    else:
                        item_seqs, item_seqs_var = None, None

                    batch_user_var = Variable(my_utils.gpu(my_utils.numpy2tensor(batch_user).type(torch.LongTensor),
                                                           use_cuda=self._use_cuda))
                    batch_item_var = Variable(my_utils.gpu(
                        my_utils.numpy2tensor(batch_item).type(torch.LongTensor),
                        use_cuda=self._use_cuda))

                    if args.adv:
                        ########################### construct adversarial pertubations:#################################
                        if args.model == 'mdr' or args.model == 'mass':
                            if args.data_type == 'upt' or args.data_type == 'pt':
                                batch_playlist_var = Variable(my_utils.gpu(
                                                                my_utils.numpy2tensor(batch_playlists).type(torch.LongTensor),
                                                                use_cuda=self._use_cuda))
                            else:
                                batch_playlist_var = None
                            if args.data_type == 'upt':
                                positive_prediction_adv = self._net(item_seqs_var,
                                                                    batch_user_var, batch_item_var, batch_playlist_var,
                                                                    adv=True)
                                negative_prediction_adv = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=True)
                            elif args.data_type == 'pt':
                                positive_prediction_adv = self._net(item_seqs_var,
                                                                    batch_playlist_var, batch_item_var, None,
                                                                    adv=True)
                                negative_prediction_adv = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=True)
                            elif args.data_type == 'ut':
                                positive_prediction_adv = self._net(item_seqs_var,
                                                                    batch_user_var, batch_item_var, None,
                                                                    adv=True)
                                negative_prediction_adv = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=True)
                            else:
                                import sys
                                print 'error, donot support data_type',args.data_type,' please select: upt, pt, or ut'
                                sys.exit(1)


                        loss_adv = self._loss_func(positive_prediction_adv, negative_prediction_adv)
                        self._optimizer.zero_grad()
                        loss_adv.backward(retain_graph=True)

                        #update adversarial noise
                        self._net._update_noise_params()

                        epoch_noise_loss += my_utils.cpu(loss_adv).data.numpy()

                        ###########################update model's params:#################################
                        if args.model == 'mdr' or args.model == 'mass':
                            if args.data_type == 'upt' or args.data_type == 'pt':
                                batch_playlist_var = Variable(my_utils.gpu(
                                                                my_utils.numpy2tensor(batch_playlists).type(torch.LongTensor),
                                                                use_cuda=self._use_cuda))
                            else:
                                batch_playlist_var = None
                            if args.data_type == 'upt':
                                positive_prediction_adv = self._net(item_seqs_var,
                                                                    batch_user_var, batch_item_var, batch_playlist_var,
                                                                    adv=True)
                                negative_prediction_adv = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=True)
                            elif args.data_type == 'pt':
                                positive_prediction_adv = self._net(item_seqs_var,
                                                                    batch_playlist_var, batch_item_var, None,
                                                                    adv=True)
                                negative_prediction_adv = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=True)
                            elif args.data_type == 'ut':
                                positive_prediction_adv = self._net(item_seqs_var,
                                                                    batch_user_var, batch_item_var, None,
                                                                    adv=True)
                                negative_prediction_adv = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=True)
                            else:
                                import sys
                                print 'error, donot support data_type',args.data_type,' please select: upt, pt, or ut'
                                sys.exit(1)

                        loss_adv = self._loss_func(positive_prediction_adv, negative_prediction_adv)

                        #normal loss
                        if args.model == 'mdr' or args.model == 'mass':
                            if args.data_type == 'upt' or args.data_type == 'pt':
                                batch_playlist_var = Variable(my_utils.gpu(
                                                                my_utils.numpy2tensor(batch_playlists).type(torch.LongTensor),
                                                                use_cuda=self._use_cuda))
                            else:
                                batch_playlist_var = None
                            if args.data_type == 'upt':
                                positive_prediction = self._net(item_seqs_var,
                                                                    batch_user_var, batch_item_var, batch_playlist_var,
                                                                    adv=False)
                                negative_prediction = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=False)
                            elif args.data_type == 'pt':
                                positive_prediction = self._net(item_seqs_var,
                                                                    batch_playlist_var, batch_item_var, None,
                                                                    adv=False)
                                negative_prediction = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=False)
                            elif args.data_type == 'ut':
                                positive_prediction = self._net(item_seqs_var,
                                                                    batch_user_var, batch_item_var, None,
                                                                    adv=False)
                                negative_prediction = self._get_neg_pred(item_seqs_var,
                                                                             batch_user_var, batch_playlist_var, batch_neg_items,
                                                                             adv=False)
                            else:
                                import sys
                                print 'error, donot support data_type',args.data_type,' please select: upt, pt, or ut'
                                sys.exit(1)

                        loss = self._loss_func(positive_prediction, negative_prediction)
                        loss_total = loss + args.reg_noise * loss_adv

                        epoch_loss += my_utils.cpu(loss_total).data.numpy()
                        self._optimizer.zero_grad()  # clear previous grad
                        loss_total.backward()

                        # clear all the grads for noises now since we are not updating the noise.
                        self._net._clear_grad(adv=True)
                        self._optimizer.step()
                        pass

                    else:

                        #sample negative items
                        #batch_neg_items = self.sample_neg_items(batch_user, num_negs=args.num_neg)
                        if args.model == 'mdr' or args.model == 'mass':
                            if args.data_type == 'upt' or args.data_type == 'pt':
                                batch_playlist_var = Variable(my_utils.gpu(
                                                                my_utils.numpy2tensor(batch_playlists).type(torch.LongTensor),
                                                                use_cuda=self._use_cuda))
                            else:
                                batch_playlist_var = None
                            if args.data_type == 'upt':
                                positive_prediction = self._net(item_seqs_var, batch_user_var, batch_item_var, batch_playlist_var)
                                negative_prediction = self._get_neg_pred(item_seqs_var, batch_user_var,
                                                                         batch_playlist_var, batch_neg_items)
                            elif args.data_type == 'pt':
                                positive_prediction = self._net(item_seqs_var, batch_playlist_var, batch_item_var, None)
                                negative_prediction = self._get_neg_pred(item_seqs_var, batch_user_var,
                                                                         batch_playlist_var, batch_neg_items)
                            elif args.data_type == 'ut':
                                positive_prediction = self._net(item_seqs_var, batch_user_var, batch_item_var, None)
                                negative_prediction = self._get_neg_pred(item_seqs_var, batch_user_var,
                                                                         batch_playlist_var, batch_neg_items)
                            else:
                                import sys
                                print 'error, donot support data_type',args.data_type,' please select: upt, pt, or ut'
                                sys.exit(1)

                        self._optimizer.zero_grad()

                        loss = self._loss_func(positive_prediction,
                                               negative_prediction
                                               )

                        #regularizer = self._net._get_l2_loss()
                        #if self._model == 'mass': loss += self._reg_mass * regularizer
                        #elif self._model == 'mdr': loss += self._reg_mdr * regularizer
                        #else: loss += self._reg_mdr * regularizer[0] + self._reg_mass * regularizer[1]

                        epoch_loss += my_utils.cpu(loss).data.numpy()

                        loss.backward()

                        self._optimizer.step()


                        if gc.DEBUG:
                            # print 'W1 :', self._net._W1.weight.data
                            # print 'W2 :', self._net._W2.weight.data
                            #
                            # print 'W1 grad:', self._net._W1.weight.grad.data
                            # print 'W2 grad:', self._net._W2.weight.grad.data
                            #
                            print 'Loss is :', loss
                            print 'Positive prediction:', positive_prediction
                            print 'negative prediction:', negative_prediction
                            if minibatch_idx >= 1: break  # debug

                epoch_loss = epoch_loss / total_interactions
                epoch_noise_loss = epoch_noise_loss/total_interactions
                t2 = time.time()
                epoch_train_time = t2 - t1
                if verbose:
                    t1 = time.time()
                    hits, ndcg = my_evaluator.evaluate(self, vadRatings, vadNegatives, topN)
                    hits_test, ndcg_test = my_evaluator.evaluate(self, testRatings, testNegatives, topN)
                    t2 = time.time()
                    eval_time = t2 - t1
                    print('|Epoch %d | Train time: %d | Train loss: %.3f | Train Noise loss: %.3f | Eval time: %d '
                          '| Vad hits@%d = %.3f | Vad ndcg@%d = %.3f '
                          '| Test hits@%d = %.3f | Test ndcg@%d = %.3f |'
                          % (epoch, epoch_train_time, epoch_loss, epoch_noise_loss, eval_time,
                             topN, hits, topN, ndcg, topN, hits_test, topN, ndcg_test))
                    if hits > best_hit or (hits == best_hit and ndcg > best_ndcg):
                    # if (hits + ndcg) > (best_hit + best_ndcg):
                    # if (ndcg > best_ndcg) or (ndcg == best_ndcg and hits > best_hit):
                        best_hit, best_ndcg, best_epoch = hits, ndcg, epoch
                        test_hit, test_ndcg = hits_test, ndcg_test
                        #self.save_checkpoint(args, best_hit, best_ndcg, epoch)
                        if args.out:
                            self.save_checkpoint(args, hits, ndcg, epoch)  # save best params

                else:
                    print('|Epoch %d | Train time: %d | loss: %.3f' % (epoch, epoch_train_time, epoch_loss))

                if np.isnan(epoch_loss) or epoch_loss == 0.0:
                    raise ValueError('Degenerate epoch loss: {}'
                                     .format(epoch_loss))

            print ('Best result: '
                   '| vad hits@%d = %.3f | vad ndcg@%d = %.3f '
                   '| test hits@%d = %.3f | test ndcg@%d = %.3f | epoch = %d' % (topN, best_hit, topN, best_ndcg,
                                                                                 topN, test_hit, topN, test_ndcg,
                                                                                 best_epoch))
    def sample_neg_items(self, user_ids, num_negs = 1):
        if num_negs == 1:
            shape = (len(user_ids), 1)
        else:
            shape = (len(user_ids), num_negs)
            # randomly sample some negative items, must improve it.
        negative_items = self._sampler.random_sample_items(
            self._n_items,
            # len(user_ids),
            shape,
            random_state=self._random_state,
            user_ids=user_ids
        )
        return negative_items

    def _get_neg_pred(self, item_seqs_var, batch_user_var, batch_playlist_var, batch_negative_items, adv=False):
        '''
        user_ids are numpy data
        :param user_ids:
        :param user_seqs:
        :return:
        '''

        negative_prediction = None
        for i in range(batch_negative_items.shape[1]):
            negative_items = batch_negative_items[:,i]


            batch_neg_item_var = Variable(my_utils.gpu(my_utils.numpy2tensor(negative_items).type(torch.LongTensor),
                                                   use_cuda=self._use_cuda))
            if self._args.data_type == 'upt':
                tmp_negative_prediction = self._net(item_seqs_var, batch_user_var, batch_neg_item_var, batch_playlist_var, adv=adv)
            elif self._args.data_type == 'pt':
                tmp_negative_prediction = self._net(item_seqs_var, batch_playlist_var, batch_neg_item_var, adv=adv)
            else: #ut
                tmp_negative_prediction = self._net(item_seqs_var, batch_user_var, batch_neg_item_var, adv=adv)
            if negative_prediction is None: negative_prediction = tmp_negative_prediction
            else: negative_prediction = torch.max(negative_prediction, tmp_negative_prediction)


        return negative_prediction

    def predict(self, user_ids, item_ids, playlist_ids):
        max_seq_len = self._max_user_seq_len
        self._net.train(False)
        #last item id is the target item

        item_seqs = self._interactions.get_batch_seqs(user_ids, item_ids, playlist_ids, max_seq_len=max_seq_len, type='all')
        item_seqs = Variable(my_utils.gpu(my_utils.numpy2tensor(item_seqs).type(torch.LongTensor),
                                          self._use_cuda), requires_grad=False)

        if self._args.data_type == 'upt':
            out = self._net(item_seqs,
                            Variable(my_utils.gpu(my_utils.numpy2tensor(user_ids), self._use_cuda)),
                            Variable(my_utils.gpu(my_utils.numpy2tensor(item_ids), self._use_cuda)),
                            Variable(my_utils.gpu(my_utils.numpy2tensor(playlist_ids), self._use_cuda)),
                            )
        elif self._args.data_type == 'ut':
            out = self._net(item_seqs,
                            Variable(my_utils.gpu(my_utils.numpy2tensor(user_ids), self._use_cuda)),
                            Variable(my_utils.gpu(my_utils.numpy2tensor(item_ids), self._use_cuda)),
                            None)
        elif self._args.data_type == 'pt':
            out = self._net(item_seqs,
                            Variable(my_utils.gpu(my_utils.numpy2tensor(playlist_ids), self._use_cuda)),
                            Variable(my_utils.gpu(my_utils.numpy2tensor(item_ids), self._use_cuda)),
                            None)
        else:
            import sys
            print 'not supporting data type:', self._args.data_type
            sys.exit(1)

        return my_utils.cpu(out).detach().data.numpy().flatten()
