import numpy as np

class W_constraint(object):

    def  __init__(self):
        print('Rotational Velocity Constraint')

    def get_margin(self,state,debug=False):
        return 1 

    def get_reward(self,state):
        return 0.0 


    def get_term_reward(self,state):
        return 0.0


        
