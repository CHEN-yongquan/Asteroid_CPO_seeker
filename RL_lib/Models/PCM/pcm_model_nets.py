import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rl_utils
class PCM_prior_vf(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_size=None, network_scale=10, activation=torch.tanh, base_lr=1e-2, cell=nn.GRUCell, 
                 recurrent_steps=1, mode='all_zero'):
        super(PCM_prior_vf, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation=activation
        self.recurrent_steps = recurrent_steps
        if hid_size is None:
            hid_size = network_scale*(act_dim+obs_dim)

        self.lr = base_lr / np.sqrt(hid_size)

        self.fc01 = nn.Linear(obs_dim, hid_size)
        self.fc02 = nn.Linear(hid_size, hid_size)

        self.fc1 = nn.Linear(obs_dim+act_dim, hid_size)
        self.rnn2 = cell(hid_size, hid_size)
        self.fc3 = nn.Linear(hid_size, hid_size)
        self.fc4 = nn.Linear(hid_size, obs_dim)
        self.fc5 = nn.Linear(hid_size, hid_size)
        self.fc6 = nn.Linear(hid_size, 1)

        self.initial_state = np.zeros((1,hid_size))
        self.initial_error = np.zeros((1,obs_dim))

        if mode == 'all_zero':
            self.zero_lb = 0
            self.zero_ub = 1
        elif mode == 'no_zero':
            self.zero_lb = 99999
            self.zero_ub = self.zero_lb + 1
        elif mode == 'rand_zero':
            self.zero_lb = 0
            self.zero_ub = self.recurrent_steps+1
        else:
            print('unsupported error zero mode')
            assert False

        print('PCM: hid size: ',hid_size)

    def get_prior(self, obs):
        obs = torch.from_numpy(obs).float()
        xp =  self.activation(self.fc01(obs))
        xp =  self.activation(self.fc02(xp))
        return xp.detach().numpy()

    def get_initial_predict(self,obs):
        r = torch.from_numpy(self.get_prior(obs))
        x = self.activation(self.fc3(r))
        x = self.fc4(x)
        v = self.activation(self.fc5(r))
        v = self.fc6(v)
        return x.detach().numpy(), np.squeeze(v.detach().numpy())

    def forward(self, obs, act,  states, errors, masks, flags, return_tensor=True,unroll=False):
        idx = np.where(flags)[0]
        zero_index = np.random.randint(self.zero_lb, self.zero_ub)
        flags = torch.from_numpy(flags).float()
        masks = torch.from_numpy(masks).float()

        obs = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()
        xp =  self.activation(self.fc01(obs))
        xp =  self.activation(self.fc02(xp))
        s[idx] = xp[idx]

        if unroll and self.recurrent_steps > 1:
            act = torch.from_numpy(act).float()
            err = torch.from_numpy(errors).float()
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  
            act = rl_utils.batch2seq(act, self.recurrent_steps)
            err = rl_utils.batch2seq(err, self.recurrent_steps)
            s = s[0] # T=0 states from rollouts
            e = err[0]

            s_outputs = []
            for i in range(self.recurrent_steps):
                if i > zero_index:
                    e = torch.zeros_like(e)
                else:
                    e = err[i]
                x = torch.cat((e,act[i]),dim=1) 
                x = self.activation(self.fc1(x))
                s = self.rnn2(x, s * masks[i])
                s_outputs.append(s)

            r = torch.stack(s_outputs)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            act = torch.from_numpy(act).float()
            err = torch.from_numpy(errors).float()

            x = torch.cat((err,act),dim=1)
            #print(err.shape, act.shape)
            x = self.activation(self.fc1(x))
            r = s = self.rnn2(x, s)
            
        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        v = self.activation(self.fc5(r))
        v = self.fc6(v)

        if return_tensor:
            return x, torch.squeeze(v), None
        else:
            return x.detach().numpy(), np.squeeze(v.detach().numpy()),  s.detach().numpy()




class PCM_vf(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_size=None, network_scale=10, activation=torch.tanh, base_lr=1e-2, cell=nn.GRUCell,
                 recurrent_steps=1, mode='all_zero'):
        super(PCM_vf, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation=activation
        self.recurrent_steps = recurrent_steps
        if hid_size is None:
            hid_size = network_scale*(act_dim+obs_dim)

        self.lr = base_lr / np.sqrt(hid_size)


        self.fc1 = nn.Linear(obs_dim+act_dim, hid_size)
        self.rnn2 = cell(hid_size, hid_size)
        self.fc3 = nn.Linear(hid_size, hid_size)
        self.fc4 = nn.Linear(hid_size, obs_dim)
        self.fc5 = nn.Linear(hid_size, hid_size)
        self.fc6 = nn.Linear(hid_size, 1)

        self.initial_state = np.zeros((1,hid_size))
        self.initial_error = np.zeros((1,obs_dim))

        if mode == 'all_zero':
            self.zero_lb = 0
            self.zero_ub = 1
        elif mode == 'no_zero':
            self.zero_lb = 99999
            self.zero_ub = self.zero_lb + 1
        elif mode == 'rand_zero':
            self.zero_lb = 0
            self.zero_ub = self.recurrent_steps+1
        else:
            print('unsupported error zero mode')
            assert False

        print('PCM no prior: hid size: ',hid_size)

    def get_prior(self, obs):
        return self.initial_state 

    def get_initial_predict(self,obs):
        r = torch.from_numpy(self.get_prior(obs)).float()
        x = self.activation(self.fc3(r))
        x = self.fc4(x)
        v = self.activation(self.fc5(r))
        v = self.fc6(v)
        return x.detach().numpy(), np.squeeze(v.detach().numpy())

    def forward(self, obs, act,  states, errors, masks, flags, return_tensor=True,unroll=False):
        zero_index = np.random.randint(self.zero_lb, self.zero_ub)
        flags = torch.from_numpy(flags).float()
        masks = torch.from_numpy(masks).float()

        obs = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        if unroll and self.recurrent_steps > 1:
            act = torch.from_numpy(act).float()
            err = torch.from_numpy(errors).float()
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)
            act = rl_utils.batch2seq(act, self.recurrent_steps)
            err = rl_utils.batch2seq(err, self.recurrent_steps)
            s = s[0] # T=0 states from rollouts
            e = err[0]

            s_outputs = []
            for i in range(self.recurrent_steps):
                if i > zero_index:
                    e = torch.zeros_like(e)
                else:
                    e = err[i]
                x = torch.cat((e,act[i]),dim=1)
                x = self.activation(self.fc1(x))
                s = self.rnn2(x, s * masks[i])
                s_outputs.append(s)

            r = torch.stack(s_outputs)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            act = torch.from_numpy(act).float()
            err = torch.from_numpy(errors).float()

            x = torch.cat((err,act),dim=1)
            x = self.activation(self.fc1(x))
            r = s = self.rnn2(x, s)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        v = self.activation(self.fc5(r))
        v = self.fc6(v)

        if return_tensor:
            return x, torch.squeeze(v), None
        else:
            return x.detach().numpy(), np.squeeze(v.detach().numpy()), s.detach().numpy()
