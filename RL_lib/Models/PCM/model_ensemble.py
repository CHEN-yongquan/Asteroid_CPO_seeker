import rl_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sklearn.utils

class Model_ensemble(object):

    """

        pred, vpred, state, error are model specific

        Looks like we actually don't use rpred and vpred in rollouots ??
        (only for env specific testing where we plot predicted versus actual trajectories)

    """

    def __init__(self, model_list):
        print('Model Ensemble:')
        self.model_list = model_list
        self.m = len(self.model_list)

        #self.keys = [model_states, model_errors]

    #model_predict,  model_vpred, model_state, model_error = model.predict(zeroed_obs, action,   nobserves[-1], model_state, model_error, np.asarray([1]), np.asarray([flag]))

    def predict(self, observe, action, next_observe, state, error, mask, flag):
        next_state_list = []
        next_error_list = []
        next_pred_list = []
        next_vpred_list = []
        state_list = np.split(state, self.m, axis=1)
        error_list = np.split(error, self.m, axis=1)
        for i in range(self.m):
            pred, vpred, state, error = self.model_list[i].predict(observe, action, next_observe, state_list[i], error_list[i], mask, flag)   
            next_pred_list.append(pred)
            next_vpred_list.append(vpred)
            next_state_list.append(state)
            next_error_list.append(error)

        return np.hstack(next_pred_list), np.hstack(next_vpred_list), np.hstack(next_state_list), np.hstack(next_error_list)

        
    def fit(self, rollouts, logger):
       
        state_list = np.split(rollouts['model_states'], self.m, axis=1)
        error_list = np.split(rollouts['model_errors'], self.m, axis=1)
        padded_state_list = np.split(rollouts['padded_model_states'], self.m, axis=1)
        padded_error_list = np.split(rollouts['padded_model_errors'], self.m, axis=1)

        for i in range(self.m):
            m_rollouts = rollouts.copy() 
            m_rollouts['model_states'] = state_list[i]
            m_rollouts['model_errors'] = error_list[i]
            m_rollouts['padded_model_states'] = padded_state_list[i]
            m_rollouts['padded_model_errors'] = padded_error_list[i]
            if False:
                print(i,' 0: ', m_rollouts['model_states'][0])
                print(i,' 1: ', m_rollouts['model_states'][1])

 
            self.model_list[i].fit(m_rollouts, logger) 

    
    def update_scalers(self, rollouts):
        for i in range(self.m):
            self.model_list[i].update_scalers(rollouts)

    def get_prior(self, unscaled_obs):
        state_list = []
        for i in range(self.m):
            state_list.append(self.model_list[i].get_prior(unscaled_obs))
        return np.hstack(state_list)

    def get_initial_error(self):
        error_list = []
        for i in range(self.m):
            error_list.append(self.model_list[i].net.initial_error)
        return np.hstack(error_list)
 
    def get_initial_predict(self, unscaled_obs):
        vpred_list = []
        pred_list = []
        for i in range(self.m):
            pred,vpred = self.model_list[i].net.get_initial_predict(unscaled_obs)
            pred_list.append(pred)
            vpred_list.append(vpred)
             
        return np.hstack(pred_list),np.hstack(vpred_list)
   
     
