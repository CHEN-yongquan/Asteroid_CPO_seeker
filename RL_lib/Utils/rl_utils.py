"""

Scaler and Logger adapted from code by Patrick Coady (pat-coady.github.io)

"""
import numpy as np
import os
import itertools
import tensorflow as tf
import torch
import torch.nn as nn
import pickle

def calc_grad_norm(parameters,norm_type=2):

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == 'inf':
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm

def grad2vec(parameters,norm_type=2):
    gvec = []
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        gvec.append(p.grad.data.detach().numpy().flatten())
    return np.hstack(gvec)

class Grad_monitor(object):
    def __init__(self, name, net):
        self.net = net
        self.norm_grads = []
        self.max_obs_ngrad = 0.
        self.max_mean = 0.
        self.max_std = 0.
        self.name = name

    def add(self,  ngrad , norm_type=2):
        if ngrad is not None:
            self.norm_grads.append(ngrad)
    def show(self):
        if len(self.norm_grads) > 0:
            self.max_obs_ngrad = np.maximum(np.max(self.norm_grads), self.max_obs_ngrad)
            self.max_mean = np.maximum(np.max(np.mean(self.norm_grads)), self.max_mean)
            self.max_std  = np.maximum(np.max(np.std(self.norm_grads)), self.max_std)

            print(self.name, ' Gradients: u/sd/Max/C Max/Max u/Max sd : %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' %  (np.mean(self.norm_grads), np.std(self.norm_grads), np.max(self.norm_grads), self.max_obs_ngrad, self.max_mean, self.max_std))
        #print('len: ',len(self.norm_grads))
        self.norm_grads = []

class Grad_monitor_old(object):
    def __init__(self, net):
        self.net = net
        self.prev_norm_grads = []
        self.norm_grads = []
        self.flat_grads = []
        self.max_obs_ngrad = 0.
        self.max_obs_fgrad = 0.
        self.max_obs_prev_ngrad = 0.

    def add(self,  prev_ngrad , norm_type=2):
        if prev_ngrad is not None:
            self.prev_norm_grads.append(prev_ngrad)
        self.norm_grads.append(calc_grad_norm(self.net.parameters()))
        self.flat_grads.append(grad2vec(self.net.parameters()))
    def show(self):
        if len(self.prev_norm_grads) > 0:
            self.max_obs_prev_ngrad = np.maximum(np.max(self.prev_norm_grads), self.max_obs_prev_ngrad)
            print('Previous Grads: Mean / SD / Max / Cum Max: %8.4f %8.4f %8.4f %8.4f' %  (np.mean(self.prev_norm_grads), np.std(self.prev_norm_grads), np.max(self.prev_norm_grads), self.max_obs_prev_ngrad))

        self.max_obs_ngrad = np.maximum(np.max(self.norm_grads), self.max_obs_ngrad)
        self.max_obs_fgrad = np.maximum(np.max(self.flat_grads), self.max_obs_fgrad)
        print('Current  Grads: Mean / SD / Max / Cum Max: %8.4f %8.4f %8.4f %8.4f' %  (np.mean(self.norm_grads), np.std(self.norm_grads), np.max(self.norm_grads), self.max_obs_ngrad))
        fgrads = np.hstack(self.flat_grads)
        print('Flat Grads:     Mean / SD / Max / Cum Max: %8.4f %8.4f %8.4f %8.4f' % (np.mean(fgrads), np.std(fgrads), np.max(fgrads), self.max_obs_fgrad)) 
        print('len: ',len(self.norm_grads), len(self.prev_norm_grads), len(fgrads))
        self.norm_grads = []
        self.prev_norm_grads = []
        self.flat_grads = []
 
         


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            #print('Scaler: ',self.means,np.sqrt(self.vars))
            self.m += n

    def apply(self,obs):
        scale = 1/(np.sqrt(self.vars) + 0.1)/3
        return (obs-self.means) * scale

    def reverse(self,obs):
        scale = 1/(np.sqrt(self.vars) + 0.1)/3
        return obs / scale + self.means

    def apply_1xsd(self,obs):
        scale = 1/(np.sqrt(self.vars) + 0.1)
        return (obs-self.means) * scale

    def reverse_1xsd(self,obs):
        scale = 1/(np.sqrt(self.vars) + 0.1)
        return obs / scale + self.means

class Image_scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim, axes=(0,2,3)):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True
        self.axes = axes

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=self.axes)
            self.vars = np.var(x, axis=self.axes)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=self.axes)
            new_data_mean = np.mean(x, axis=self.axes)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def apply(self,obs):
        channels = []
        for c in range(obs.shape[1]):
            scale = 1/(np.sqrt(self.vars[c]) + 0.1)/3
            means = self.means[c]
            channels.append( (obs-means) * scale)
        return np.stack(channels,axis=-3)

    def reverse(self,obs):
        for c in range(obs.shape[1]):
            scale = 1/(np.sqrt(self.vars[c]) + 0.1)/3
            means = self.means[c]
            channels.append(obs / scale + means)
        return np.stack(channels,axis=-3)
 

class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """

        self.write_header = True
        self.log_entry = {}
        self.writer = None  # DictWriter created with first call to write() method
        self.scores = []

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry)

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {}, Mean R = {:.1f}  Std R = {:.1f}  Min R = {:.1f}'.format(log['_Episode'],
                                                               log['_MeanReward'], log['_StdReward'], log['_MinReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                #print(key, log[key])
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        pass


def discretize(x,n,min_action,max_action):
    """
        n is number of samples per dimension
        d is the dimension of x

    """
    x = np.clip(x,min_action,max_action)
    bins = np.linspace(min_action,max_action,n+1)
    #print(bins)
    indices = np.digitize(x,bins) - 1
    #print(indices)
    idx = indices >= n
    indices[idx] = n-1
    return indices 

class Action_converter(object):
    def __init__(self,action_dim,actions_per_dim,min_action=-1.,max_action=1.):
        self.action_dim = action_dim
        self.actions_per_dim = actions_per_dim
        self.min_action = min_action
        self.max_action = max_action
        self.actions = np.linspace(min_action, max_action, actions_per_dim)
        self.action_table = np.asarray(list(itertools.product(self.actions, repeat=action_dim)))
        print(self.action_table)
    def idx2action_old(self,idx):
        return self.action_table[idx].T

    def idx2action(self,idx):
        action = np.squeeze(self.action_table[idx])
        if len(action.shape) == 1:
            action = np.expand_dims(action,axis=0)
        return action
    
    def action2idx(self,action):
        idx = discretize(action,self.actions_per_dim,self.min_action,self.max_action)
        return idx

class Action_converter_old(object):
    def __init__(self,action_dim,actions_per_dim,min_action=-1.,max_action=1.):
        self.action_dim = action_dim
        self.actions_per_dim = actions_per_dim
        self.min_action = min_action
        self.max_action = max_action
        self.actions = np.linspace(min_action, max_action, actions_per_dim)
        self.action_table = np.asarray(list(itertools.product(self.actions, repeat=action_dim)))
        print(self.action_table)
    def idx2action(self,idx):
        return self.action_table[idx].T

    def action2idx(self,action):
        idx = discretize(action,self.actions_per_dim,self.min_action,self.max_action) 
        return idx


def shuffle_list_by_chunks(data_list, T):
    m = data_list[0].shape[0]
    assert m % T == 0
    idx = np.random.permutation(m//T)
    out = []
    for d in data_list:
        assert d.shape[0] == m
        out.append(shuffle_by_chunks(d,T,idx))
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

def shuffle_by_chunks(data,T,idx):
    m = data.shape[0]
    split_data = np.split(data,m//T)
    split_data = np.asarray(split_data)
    shuffled_data = split_data[idx]
    shuffled_data = np.concatenate(shuffled_data)
    return shuffled_data 
 
def calc_exp_var(sdr, vpred, mask, unpad):
    # diagnose over-fitting of val func
    if unpad:
        sdr,vpred = unpad_list([sdr,vpred], mask)
    exp_var = 1 - np.var(sdr - vpred) / np.var(sdr)  
    return exp_var

def unpad(x, masks):
    idx = np.where(masks)[0]
    x = x[idx]
    return x

def unpad_list(X, masks):
    out = []
    for x in X:
        s1 = x.shape
        idx = np.where(masks)[0]
        out.append(x[idx])
    return tuple(out) 


def pad(data,start_flags,T):
    
    starts = np.where(start_flags)[0]
    ends = np.hstack((starts[1:],data.shape[0]))
    D = []
    M = []
    for i in range(starts.shape[0]):
        chunk = data[starts[i]:ends[i]]
        #print(i,starts[i],ends[i])
        #print(i,chunk.shape[0],chunk)
        d,m = add_padding(chunk,T)
        #print(i,d)
        D.append(d)
        M.append(m)
    return np.vstack(D),np.hstack(M)

def add_padding(data, recurrent_steps) :
    data_old = data.copy()
    rem = data.shape[0] % recurrent_steps
    if rem is not 0:
        extra = recurrent_steps - rem
    else:
        extra = 0 
    mask = np.ones(len(data)+extra)
    if extra > 0:
        mask[-extra:] = 0
    if  isinstance(data[0], np.ndarray): 
        #print('foo: ',data[0], type(data[0]))
        #data = np.vstack((data,  np.zeros((extra, data.shape[1])) ))
        foo = [extra]
        for tmp in data.shape[1:]:
            foo.append(tmp)
        foo = tuple(foo)
        data = np.vstack( (data,  np.zeros(foo, dtype=type(data.flatten()[0]))) )
    else:
        data = np.hstack((data,  np.zeros(extra) ))
    if not ( data.shape[0] % recurrent_steps == 0):
        print(data.shape, data_old.shape, recurrent_steps, extra)
        assert False
    return(data, mask)

def batch2seq_np(x, T):
    m = x.shape[0] // T
    x = np.reshape(x, (m,T,-1)) 
    x = np.transpose(x,  (1,0,2))
    return x

def batch2seq(x, T):
    m = x.size(0) // T
    x = torch.transpose(x.view(m,T,-1), 0,1)
    return x

def seq2batch(x,T):
    m = x.size(1)*T
    n = x.size(2)
    #x = torch.transpose(x,0,1).view(m,n)
    x = torch.transpose(x,0,1).contiguous().view(m,n)
    return x


def get_mini_ids(m,k):
    num_batches = max(m // k, 1)
    batch_size = m // num_batches
    last_batch_size = m % num_batches
    indices = []
    for j in range(num_batches):
        start = j * batch_size
        end = (j + 1) * batch_size
        indices.append([start,end])
    if last_batch_size > 0:
        start=end
        end = m
        indices.append([start,end])
    return indices


def calc_masked_loss(pred,targ,mask):
    mask = torch.from_numpy(mask).float()
    error = pred - targ
    loss = torch.sum(torch.mul(error, error),dim=1)
    loss = mask * loss 
    loss = torch.sum(loss) / (torch.sum(mask)*pred.size(1))
    return loss

def calc_masked_loss_from_error(error,mask):
    mask = torch.from_numpy(mask).float()
    loss = torch.sum(torch.mul(error, error),dim=1)
    loss = mask * loss
    loss = torch.sum(loss) / (torch.sum(mask)*pred.size(1))
    return loss


def xn_init(m):
    if isinstance(m,nn.Linear):
        print('\txn_init: layer ',m) 
        torch.nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.GRUCell) or isinstance(m,nn.RNNCell) or isinstance(m,nn.LSTMCell):
        print('\txn_init: layer ',m)
        torch.nn.init.xavier_normal_(m.weight_ih.data)
        torch.nn.init.xavier_normal_(m.weight_hh.data)
        m.bias_ih.data.fill_(0)
        m.bias_hh.data.fill_(0)

def ortho_init(m):
    if isinstance(m,nn.Linear):
        print('\tortho init: layer ',m)
        torch.nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.GRUCell) or isinstance(m,nn.RNNCell) or isinstance(m,nn.LSTMCell):
        print('\tortho init: layer ',m)
        torch.nn.init.xavier_normal_(m.weight_ih.data)
        torch.nn.init.xavier_normal_(m.weight_hh.data)
        m.bias_ih.data.fill_(0)
        m.bias_hh.data.fill_(0)

def xu_init(m):
    if isinstance(m,nn.Linear):
        print('\txu_init: layer ',m)
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.GRUCell) or isinstance(m,nn.RNNCell) or isinstance(m,nn.LSTMCell):
        print('\txu_init: layer ',m)
        torch.nn.init.xavier_uniform_(m.weight_ih.data)
        torch.nn.init.xavier_uniform_(m.weight_hh.data)
        m.bias_ih.data.fill_(0)
        m.bias_hh.data.fill_(0)

def n_init(m):
    if isinstance(m,nn.Linear):
        print('\tn_init: layer ',m)
        torch.nn.init.normal_(m.weight.data,std=1/np.sqrt(m.weight.data.size()[1]))
        m.bias.data.fill_(0)
    elif isinstance(m,nn.GRUCell) or isinstance(m,nn.RNNCell) or isinstance(m,nn.LSTMCell):
        print('\tn_init: layer ',m)
        torch.nn.init.normal_(m.weight_ih.data)
        torch.nn.init.normal_(m.weight_hh.data)
        m.bias_ih.data.fill_(0)
        m.bias_hh.data.fill_(0)

def default_init(m):
    return xn_init(m)

def weights_init_orig(m):
    print('\t Old Init')

class RBF_layer(object):
    def __init__(self, dim, centers_per_dim=2, sigma=None):
        self.dim = dim
        self.centers_per_dim = centers_per_dim
        d = 1 - 1. / centers_per_dim
        centers_1d = np.linspace(-d, d, centers_per_dim)
        self.centers = np.asarray(list(itertools.product(centers_1d, repeat=dim)))
        self.sigma = np.sqrt(np.linalg.norm(np.ones(dim))) 
        print('RBF: ', self.centers)
 
    def forward_old(self, x):
        activations = np.exp(-np.linalg.norm(self.centers - x,axis=1)**2/(2*self.sigma**2)) 
        return activations

    def forward(self, x):
        activations = []
        for c in self.centers:
            a = np.exp(-np.linalg.norm(c - x,axis=1)**2/(2*self.sigma**2))
            activations.append(a)
        return np.asarray(activations).T

    def size(self):
        return self.centers.shape[0]
             
