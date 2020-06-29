import rl_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import env_utils as envu
import matplotlib.pyplot as plt
from time import time
import sklearn.utils

class Model(object):

    def __init__(self, net, epochs=20,  shuffle=False, max_grad_norm=999, p_coeff=0.85 , cliprange=0.5, test_pred=False, 
                 init_func=rl_utils.default_init, test_steps=1, rollout_limit=1, print_every=10, print_skip=5, verbose=False,
                 model_idx=None, learn_rewards=False, use_deltas=True):
        print('PCM Model1:')
        net.apply(init_func)
        self.net = net
        if self.net.recurrent_steps > 1:
            self.use_padding = True
        else:
            self.use_padding = False
        self.use_deltas = use_deltas
        self.model_idx = model_idx
        self.test_pred = test_pred
        self.learn_rewards = learn_rewards
        self.epochs = epochs
        self.shuffle = shuffle
        self.max_grad_norm = max_grad_norm
        self.cliprange = cliprange
        self.p_coeff = p_coeff
        self.vf_coeff = 1. - p_coeff
        assert self.vf_coeff >= 0 and self.vf_coeff <= 1
        self.test_steps = test_steps
        self.print_every = print_every
        self.print_skip = print_skip
        self.verbose = verbose
        self.nobs_scaler = rl_utils.Scaler(net.obs_dim)
        self.obs_scaler = rl_utils.Scaler(net.obs_dim)
        self.act_scaler = rl_utils.Scaler(net.act_dim)
        self.delta_scaler = rl_utils.Scaler(net.obs_dim)

        self.max_obs_grad = 0.0
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.net.lr)

        self.grad_monitor = rl_utils.Grad_monitor('Model ',net)

        self.rollout_list = []
        self.rollout_limit = rollout_limit

        self.count = 0
        self.logged_pred_errors = {}
        self.logged_vpred_errors = {}
        
        print('\tShuffle :          ',self.shuffle)
        print('\tMax Grad Norm:     ',self.max_grad_norm)
        print('\tRecurrent Steps:   ',self.net.recurrent_steps)
        print('\tPredict Coeff :    ',self.p_coeff)
        print('\tValue   Coeff :    ',self.vf_coeff)
        print('\tRollouts / Update: ',self.rollout_limit)

    def save_params(self,fname):
        fname = 'model_' + fname + '.pt'
        param_dict = {}
        param_dict['nobs_scaler_u']  = self.nobs_scaler.means
        param_dict['nobs_scaler_var'] = self.nobs_scaler.vars
        param_dict['obs_scaler_u']   = self.obs_scaler.means
        param_dict['obs_scaler_var']  = self.obs_scaler.vars
        param_dict['act_scaler_u']  = self.act_scaler.means
        param_dict['act_scaler_var'] = self.act_scaler.vars
        param_dict['delta_scaler_u']  = self.delta_scaler.means
        param_dict['delta_scaler_var'] = self.delta_scaler.vars
        param_dict['net_state'] = self.net.state_dict()
        torch.save(param_dict, fname)

    def load_params(self,fname):
        fname = 'model_' + fname + '.pt'
        param_dict = torch.load(fname)
        self.nobs_scaler.means = param_dict['nobs_scaler_u']
        self.nobs_scaler.vars = param_dict['nobs_scaler_var']
        self.obs_scaler.means = param_dict['obs_scaler_u']
        self.obs_scaler.vars = param_dict['obs_scaler_var']
        self.act_scaler.means = param_dict['act_scaler_u']
        self.act_scaler.vars = param_dict['act_scaler_var']
        self.delta_scaler.means = param_dict['delta_scaler_u']
        self.delta_scaler.vars = param_dict['delta_scaler_var']
        self.net.load_state_dict(param_dict['net_state'])
 
 
    def update_scalers(self, rollouts): 
        self.nobs_scaler.update(rollouts['vector_nobserves'])
        self.obs_scaler.update(rollouts['vector_observes'])
        self.act_scaler.update(rollouts['env_actions'])
        self.delta_scaler.update(rollouts['vector_nobserves'] - rollouts['vector_observes'])
 
    def fit(self, rollouts, logger):
        if len(self.rollout_list) == self.rollout_limit:
            del self.rollout_list[0]
        self.rollout_list.append(rollouts)
        keys = self.rollout_list[0].keys()
        comb_rollouts = {}
        for k in keys:
            comb_rollouts[k] = np.concatenate([r[k] for r in self.rollout_list])
        self.fit1(comb_rollouts, logger)
 
    def fit1(self, rollouts, logger):

        if self.use_padding:
            key = 'padded_'
        else:
            key = ''

        unscaled_obs = rollouts[key + 'vector_observes']
        unscaled_act = rollouts[key + 'env_actions']
        unscaled_nobs = rollouts[key + 'vector_nobserves']

        states =        rollouts[key + 'model_states']
        errors =        rollouts[key + 'model_errors']
        masks =         rollouts[key + 'masks']
        flags =         rollouts[key + 'flags']
        if self.learn_rewards:
            sdr = rollouts[key + 'rewards1'] #+  rollouts[key + 'rewards2']
        else:
            sdr =           rollouts[key + 'disc_sum_rew']

        obs = self.obs_scaler.apply(unscaled_obs)
        act = self.act_scaler.apply(unscaled_act)
        targets = self.delta_scaler.apply(unscaled_nobs - unscaled_obs)

        if self.test_pred:
            self.test_lt(unscaled_obs, unscaled_act,  unscaled_nobs, states, errors, masks, flags, targets, sdr,  self.test_steps)

        t0 = time()

        # calculate explained variance before
        pred ,vpred_old,  _ = self.net.forward(obs, act,  states, errors, masks, flags,  return_tensor=False)
        exp_var_old = rl_utils.calc_exp_var(sdr, vpred_old, masks, self.use_padding)
       
        loss_old = self.get_loss1_np(pred,targets,masks)

        for e in range(self.epochs):
            if self.shuffle:
                obs, act, targets, states, errors, masks, flags, sdr,  vpred_old   =  \
                        sklearn.utils.shuffle(obs, act, targets, states, errors, masks, flags, sdr,  vpred_old )
            self.optimizer.zero_grad()
            pred, vpred,  _  = self.net.forward(obs, act,  states, errors, masks, flags, unroll=True)
            targs = torch.from_numpy(targets).float()
            vtargs = torch.from_numpy(sdr).float()
            p_loss = self.get_loss1(pred, targs, masks) 
            vf_loss = self.get_loss2(vpred,  torch.from_numpy(vpred_old).float(), vtargs, masks)
            loss = self.p_coeff * p_loss + self.vf_coeff * vf_loss 
            loss.backward()
            if self.max_grad_norm is not None:
                ng = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            else:
                ng = None
            self.grad_monitor.add(ng)

            if self.max_grad_norm is not None:
                #print('Clipping Grads')
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm) 
            self.optimizer.step()

        self.grad_monitor.show()
        t1 = time()
        if self.verbose:
            print('MODEL SDR: ',np.mean(sdr),np.max(np.abs(sdr)))
            print('Model FIT: ',np.mean(np.abs(obs)), np.mean(np.abs(targets)),  np.mean(np.abs(act)))
            print('Model FIT: ',t1-t0,obs.shape)
        # calculate explained variance
        _, vpred,  _ = self.net.forward(obs, act,  states, errors, masks, flags,  return_tensor=False)
        exp_var_new = rl_utils.calc_exp_var(sdr, vpred, masks, self.use_padding)
 
        logger.log({'Model_' + str(self.model_idx) +  ' P Loss New': p_loss.detach().numpy(),
                    'Model_' + str(self.model_idx) +  ' P Loss Old': loss_old,
                    'Model_' + str(self.model_idx) +  ' VF Loss': vf_loss.detach().numpy(),
                    'Model_' + str(self.model_idx) +  ' ExpVarOld': exp_var_old,
                    'Model_' + str(self.model_idx) +  ' ExpVarNew': exp_var_new})
                    

    def get_loss1_np(self,pred,targ,masks):
        if self.use_padding:
            pred,targ = rl_utils.unpad_list([pred,targ],masks)
        error = pred - targ
        loss = 0.5 * np.mean(error * error)
        return loss

    def get_loss1(self,pred,targ,masks):
        if self.use_padding:
            pred,targ = rl_utils.unpad_list([pred,targ],masks)
        error = pred - targ
        loss = 0.5 * torch.mean(torch.mul(error, error))
        return loss

    def get_loss2(self, pred, old_pred, targ, masks):
        if self.use_padding:
            pred, old_pred, targ = rl_utils.unpad_list([pred, old_pred, targ],masks)
        if self.cliprange is not None:
            vpred_clipped = old_pred + torch.clamp(pred - old_pred, -self.cliprange, self.cliprange)
            error = pred - targ
            loss1 = torch.mul(error, error)
            error = vpred_clipped - targ
            loss2 = torch.mul(error, error)
            loss = 0.5 * torch.mean(torch.max(loss1,loss2))
        else:
            error = pred - targ
            loss = 0.5 * torch.mean(torch.mul(error, error))
        return loss

    def get_prior(self, unscaled_obs):
        obs = self.obs_scaler.apply(unscaled_obs)
        prior = self.net.get_prior(obs)
        return prior

    def get_initial_predict(self, unscaled_obs, scale_outputs=True):
        obs = self.obs_scaler.apply(unscaled_obs)
        pred,vpred = self.net.get_initial_predict(obs)
        if scale_outputs:
            delta = self.delta_scaler.reverse(pred)
            pred = unscaled_obs + delta
        else:
            pred = pred + obs
        return pred,vpred 
 
    def predict(self, unscaled_obs, unscaled_act,  unscaled_nobs, states, errors, masks, flags, scale_outputs=True):
        obs = self.obs_scaler.apply(unscaled_obs)
        nobs = self.nobs_scaler.apply(unscaled_nobs)
        delta_actual = nobs - obs
        act = self.act_scaler.apply(unscaled_act)
        pred, vpred,  next_state = self.net.forward(obs, act,   states, errors, masks, flags, return_tensor=False)
        targ = nobs
        next_error = pred - delta_actual 
        if scale_outputs:
            delta = self.delta_scaler.reverse(pred)
            pred = unscaled_obs + delta
        else:
            pred = pred + obs

        return pred,  vpred, next_state, next_error

    def pad2seq(self, x , flags, masks, steps):
        x = rl_utils.unpad(x,masks)
        flags = rl_utils.unpad(flags,masks)
        x,_ = self.pad(x,flags, steps)
        x = rl_utils.batch2seq_np(x, steps)
        return x

    def pad(self, data,start_flags,T):
        starts = np.where(start_flags)[0]
        ends = np.hstack((starts[1:],data.shape[0]))
        D = []
        M = []
        for i in range(starts.shape[0]):
            chunk = data[starts[i]:ends[i]]
            d,m = rl_utils.add_padding(chunk,T)
            D.append(d)
            M.append(m)
        if len(data.shape) > 1:
            return np.vstack(D),np.hstack(M)
        else:
            return np.hstack(D),np.hstack(M)

    def test_lt(self, unscaled_obs, unscaled_act,  unscaled_nobs, states, errors, masks , flags, targ, sdr,  steps):
        flags1 = flags.copy()
        masks1 = masks.copy()
        unscaled_obs =  self.pad2seq(unscaled_obs,  flags1, masks1, steps) 
        unscaled_act =  self.pad2seq(unscaled_act,  flags1, masks1, steps) 
        unscaled_nobs = self.pad2seq(unscaled_nobs, flags1, masks1, steps) 
        states =        self.pad2seq(states,        flags1, masks1, steps)        
        errors =        self.pad2seq(errors,        flags1, masks1, steps)        
        targ =          self.pad2seq(targ,          flags1, masks1, steps)
        masks =         self.pad2seq(masks,         flags1, masks1, steps)
        sdr =           self.pad2seq(sdr,           flags1, masks1, steps)
        flags =         self.pad2seq(flags,         flags1, masks1, steps)
        if self.verbose:
            print('MASK1: ',masks1.shape,np.sum(masks1))
            print(unscaled_obs.shape,flags1.shape,masks1.shape,steps)
            print('MASK: ',masks.shape,np.sum(masks))
            print('STATES: ',states.shape) 
        e = errors[0]
        s = states[0]
        pred = unscaled_obs[0]
        for t in range(steps):
            # returns pred, next state, next error
            pred, vpred, s, _ = self.predict(pred, unscaled_act[t],  unscaled_nobs[t], s, e, masks[t], flags[t], scale_outputs=False)
            e = np.zeros_like(e)  # for PCM, error of zero is same as feeding in last obs        
            #d = np.abs(self.nobs_scaler.apply(pred) - self.nobs_scaler.apply(targ[t]))
            d = np.abs(pred - targ[t])

            d = rl_utils.unpad(d,masks[t])
            ev = rl_utils.calc_exp_var(np.squeeze(sdr[t]), vpred, masks[t], self.use_padding)
            if t % self.print_skip == 0 or t == steps-1:
                if not t in self.logged_pred_errors:
                    self.logged_pred_errors[t] = []
                    self.logged_vpred_errors[t] = []
                self.logged_pred_errors[t].append(d)
                self.logged_vpred_errors[t].append(ev)

 
        if self.count % self.print_every == 0:
            for t in range(steps):
                if t % self.print_skip == 0 or t == steps-1:
                    d = self.logged_pred_errors[t]
                    d = np.concatenate(d)
                    if d.shape[0] > 0: # new
                        print('Model P Errors (t, mean, std, max): %4d %8.4f %8.4f %8.4f' % (t,np.mean(d), np.std(d), np.max(d) ) )
            for t in range(steps):
                if t % self.print_skip == 0 or t == steps-1:
                    d = self.logged_vpred_errors[t]
                    if len(d) > 0: # new
                        print('Model EV       (t, mean, std, min): %4d %8.4f %8.4f %8.4f' % (t,np.mean(d), np.std(d), np.min(d) ) )

            f = '{:7.3f}'
            total_errors = []
            for k,v in self.logged_pred_errors.items():
                total_errors.append(np.concatenate(v))
            total_errors = np.concatenate(total_errors)

            print('Model P Mean Errors: ' + envu.print_vector(' |',np.mean(total_errors,axis=0),f))
            print('Model P Std  Errors: ' + envu.print_vector(' |',np.std(total_errors,axis=0),f))
            print('Model P Max  Errors: ' + envu.print_vector(' |',np.max(total_errors,axis=0),f))

            total_errors = []
            for k,v in self.logged_vpred_errors.items():
                total_errors.append(v)
            total_errors = np.concatenate(total_errors)
            print('Model EV Mean: ' + envu.print_vector(' |',np.mean(total_errors,axis=0),f))
            print('Model EV Std: ' + envu.print_vector(' |',np.std(total_errors,axis=0),f))
            print('Model EV Min: ' + envu.print_vector(' |',np.min(total_errors,axis=0),f))

            for k,v in self.logged_pred_errors.items():
                self.logged_pred_errors[k] = []
                self.logged_vpred_errors[k] = []

        self.count += 1
