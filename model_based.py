import glob
import os
import torch

class ModelBased(object):
    def __init__(self):
        self._net = None
        self._optimizer = None

    def _get_model_name(self, args):
        return str(args.model)
    def _make_model_desc(self, args, model = ''):
        # model_desc = '%s_'%args.model #mdr, mass, or masr
        if args.model == 'mass' or args.model == 'mdr':
            model_desc = 'type-%s_'%args.data_type
        else:
            #combined model
            if model == 'mass':
                model_desc = 'type-%s_' % args.data_type_mass
            else:
                model_desc = 'type-%s_' % args.data_type_mdr
        model_desc += 'nfactors_%d' % args.num_factors
        model_name = args.model if model == '' else model
        if model_name == 'mdr':
            model_desc += '_reg_%s' % str(args.reg_mdr)
            model_desc += '_act_%s'% str(args.act_func_mdr)
            # print model_desc
            return model_desc
        elif model_name == 'mass':
            model_desc += '_reg_%s' % str(args.reg_mass)
            model_desc += '_seq%d' % args.max_seq_len
            model_desc += '_act_%s'% str(args.act_func)
            return model_desc
        elif model_name == 'masr':
            model_desc += '_mdr-reg_%s' % str(args.reg_mdr)
            model_desc += '_mass-reg_%s' % str(args.reg_mass)
            model_desc += '_seq%d' % args.max_seq_len
            model_desc += '_act_%s' % str(args.act_func)
            return model_desc


    def save_checkpoint(self, args, best_hit, best_ndcg, epoch):
        model_state_dict = self._net.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': self._optimizer.state_dict(),
            'settings': args,
            'best_hits': best_hit, #best validation/development hit
            'best_ndcg': best_ndcg, #best validation/development ndcg
            'epoch': epoch}
        if args.adv:
            model = 'adv-' + args.model
        else:
            model = args.model
        model_name = '%s_%s_%s_hits_%.3f_ndcg_%.3f.chkpt' % (args.dataset,
                                                             model,
                                                             self._make_model_desc(args),
                                                             best_hit, best_ndcg)
        model_path = os.path.join(args.saved_path, model_name)
        torch.save(checkpoint, model_path)


    def load_checkpoint(self, args):
        lst_models = ['mass', 'mdr']
        if args.eval:
            lst_models = ['mass', 'mdr', 'masr']
        if args.model in lst_models:
            best_hits = 0.0
            best_ndcg = 0.0
            best_saved_file = ''

            saved_file_pattern = '%s_%s_%s*'%(args.dataset, args.model, self._make_model_desc(args))
            print 'searching: ',saved_file_pattern
            for filepath in glob.glob(os.path.join(args.saved_path,saved_file_pattern)):
                # filepath = os.path.join(args.saved_path, fname)
                if os.path.isfile(filepath):
                    print("=> loading checkpoint '{}'".format(filepath))
                    checkpoint = torch.load(filepath)
                    hits = float(checkpoint['best_hits'])
                    ndcg = float(checkpoint['best_ndcg'])
                    #if hits > best_hits or (hits == best_hits and ndcg > best_ndcg):
                    if ndcg > best_ndcg or (ndcg == best_ndcg and hits > best_hits):
                        best_saved_file=filepath
                        best_hits = hits
                        best_ndcg = ndcg
            if best_saved_file != '':
                checkpoint = torch.load(best_saved_file)

                # load only parameters that are available
                model_dict = self._net.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                self._net.load_state_dict(model_dict)

                ######### no need to load state dict for optimizer
                # self._optimizer.load_state_dict(checkpoint['optimizer'])

                print("=> loaded checkpoint '{}' (epoch {})"
                              .format(best_saved_file, checkpoint['epoch']))

            return (best_hits, best_ndcg)

        else:
            ######## MASR model #######
            #load best mdr checkpoint
            best_mdr_file = ''
            best_mdr_hits, best_mdr_ndcgs = 0,0
            mdr_files_pattern = '%s_mdr_%s*'%(args.dataset, self._make_model_desc(args, 'mdr'))
            # print mdr_files_pattern
            print 'searching mdr: ',mdr_files_pattern
            for filepath in glob.glob(os.path.join(args.saved_path, mdr_files_pattern)):
                print filepath
                if os.path.isfile(filepath):
                    print("=> loading checkpoint '{}'".format(filepath))
                    if args.cuda:
                        checkpoint = torch.load(filepath)
                    else:
                        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
                    hits = float(checkpoint['best_hits'])
                    ndcg = float(checkpoint['best_ndcg'])
                    #if hits > best_mdr_hits or (hits == best_mdr_hits and ndcg > best_mdr_ndcgs):
                    if ndcg > best_mdr_ndcgs or (ndcg == best_mdr_ndcgs and hits > best_mdr_hits):
                        best_mdr_file=filepath
                        best_mdr_hits = hits
                        best_mdr_ndcgs = ndcg

            #load best mass checkpoint
            best_mass_file = ''
            best_mass_hits, best_mass_ndcgs = 0, 0
            mass_files_pattern = '%s_mass_%s*' % (args.dataset, self._make_model_desc(args, 'mass'))
            print 'searching mass: ',mass_files_pattern
            for filepath in glob.glob(os.path.join(args.saved_path, mass_files_pattern)):
                if os.path.isfile(filepath):
                    print("=> loading checkpoint '{}'".format(filepath))
                    if args.cuda:
                        checkpoint = torch.load(filepath)
                    else:
                        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

                    hits = float(checkpoint['best_hits'])
                    ndcg = float(checkpoint['best_ndcg'])
                    #if hits > best_mass_hits or (hits == best_mass_hits and ndcg > best_mass_ndcgs):
                    if ndcg > best_mass_ndcgs or (ndcg == best_mass_ndcgs and hits > best_mass_hits):
                        best_mass_file = filepath
                        best_mass_hits = hits
                        best_mass_ndcgs = ndcg


            #now loading best checkpoints from mdr and mass for masr:
            if best_mdr_file != '':
                #load checkpoint into cpu
                if args.cuda:
                    checkpoint = torch.load(best_mdr_file)
                else:
                    checkpoint = torch.load(best_mdr_file, map_location=lambda storage, loc: storage)


                self._net._mdr.load_state_dict(checkpoint['model'])




                # self._optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                                .format(best_mdr_file, checkpoint['epoch']))
              
                # no train the mass and mdr?
                for params in self._net._mdr.parameters():
                    params.requires_grad = False

            if best_mass_file != '':
                #load checkpoint into cpu:
                if args.cuda:
                    checkpoint = torch.load(best_mass_file)
                else:
                    checkpoint = torch.load(best_mass_file, map_location=lambda storage, loc: storage)

                self._net._mass.load_state_dict(checkpoint['model'])


                print("=> loaded checkpoint '{}' (epoch {})"
                                 .format(best_mass_file, checkpoint['epoch']))
                #no train the mass and mdr?
                for params in self._net._mass.parameters():
                    params.requires_grad = False


            return 0,0




