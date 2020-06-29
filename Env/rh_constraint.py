import numpy as np

class RH_constraint(object):

    def  __init__(self,  terminate_on_violation=True, 
                  rh_limit=100, rh_penalty=-50., debug=False):
        self.rh_limit = rh_limit
        self.rh_penalty = rh_penalty
        self.terminate_on_violation = terminate_on_violation
        print('Position Hysterises Constraint')
        self.min_range = None
        self.cnt = 0
        self.vio_cnt = 0
        self.debug = debug

    def reset(self):
        self.min_range = 1e12

    def step(self, state):
        self.min_range = np.minimum(self.min_range, np.linalg.norm(state['position']))
        if self.debug:
            print(np.linalg.norm(state['position']), self.min_range)

    def get_margin(self,state,debug=False):
        rh = np.linalg.norm(state['position']) - self.min_range
        if rh > self.rh_limit: 
            margin = -1
        else:
            margin = 1
        return margin 

    def get_term_reward(self,state):
        margin = self.get_margin(state)
        if margin < 0:
            self.cnt += 1
            self.vio_cnt += 1
            if self.cnt % 100 == 0:
                 print('*** RH VIO  CNT: ',self.vio_cnt)
            return self.rh_penalty 
        else:
            return 0.0


        
