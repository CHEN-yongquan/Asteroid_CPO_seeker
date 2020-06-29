import numpy as np
from time import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import env_utils as envu
import pylab
import matplotlib
import render_traj_sensor

class Env(object):
    def __init__(self, ic_gen, lander, dynamics, logger,
                 landing_site_range=0.0,
                 render_func=render_traj_sensor.render_traj,  
                 glideslope_constraint=None,
                 attitude_constraint=None,
                 debug_steps=False,
                 w_constraint=None,
                 rh_constraint=None,
                 reward_object=None,
                 debug_done=False,
                 nav_period=10,
                 tf_limit=5000.0, allow_plotting=True, print_every=1,):
        self.landing_site_range = landing_site_range
        self.nav_period = nav_period
        self.debug_done = debug_done
        self.debug_steps = debug_steps 
        self.logger = logger
        self.lander = lander
        self.rl_stats = RL_stats(lander,logger,render_func, print_every=print_every,allow_plotting=allow_plotting) 
        self.tf_limit = tf_limit
        self.display_errors = False
        self.dynamics = dynamics 
        self.allow_plotting = allow_plotting
        self.ic_gen = ic_gen 
        self.episode = 0
        self.glideslope_constraint = glideslope_constraint
        self.attitude_constraint = attitude_constraint
        self.w_constraint = w_constraint
        self.rh_constraint = rh_constraint
        self.reward_object = reward_object

        self.terminate_on_range = landing_site_range > 0.01
        if allow_plotting:
            plt.clf()
            plt.cla()
        print('lander env RHL')
        
    def reset(self): 
        self.ic_gen.set_ic(self.lander, self.dynamics)
        self.glideslope_constraint.reset(self.lander.state)
        self.lander.sensor.reset(self.lander.state)
        self.rh_constraint.reset() 
        self.steps = 0
        self.t = 0.0

        self.lander.clear_trajectory()
        agent_image, agent_state = self.lander.get_state_agent(self.t)
        self.lander.update_trajectory(False, self.t)
        return agent_image, agent_state 

    def check_for_done(self,lander):
        done = False
        vc = envu.get_vc(lander.state['position'], lander.state['velocity'])
        if self.glideslope_constraint.get_margin() < 0.0 and self.glideslope_constraint.terminate_on_violation:  
            done = True
            if self.debug_done:
                print('Glideslope: ', self.glideslope_constraint.get_margin() , self.steps)

        if self.attitude_constraint.get_margin(lander.state) < 0.0 and self.attitude_constraint.terminate_on_violation:
            done = True
            if self.debug_done:
                print('Attitude: ', self.attitude_constraint.get_margin(lander.state) , self.steps)

        if self.w_constraint.get_margin(lander.state) < 0.0 and self.w_constraint.terminate_on_violation:
            done = True
            if self.debug_done:
                print('Rot Vel:  ', self.w_constraint.get_margin(lander.state) , self.steps)

        if self.rh_constraint.get_margin(lander.state) < 0.0 and self.rh_constraint.terminate_on_violation:
            done = True
            if self.debug_done:
                print('RH Const: ', self.rh_constraint.get_margin(lander.state) , self.steps)

        rel_pos = lander.state['position'] - lander.asteroid.landing_site
        pos_dvec = rel_pos / np.linalg.norm(rel_pos)
        if self.terminate_on_range:
            if np.linalg.norm(rel_pos) < self.landing_site_range: 
                done = True
        else:
            if  np.dot(pos_dvec, self.lander.asteroid.landing_site_pn) < 0.0: 
                done = True
        if self.t > self.tf_limit:
            done = True
            if self.debug_done:
                print('Timeout: ', self.steps)
        #print('pos: ', lander.state['position'], rel_pos)
        #print('vel: ', lander.state['velocity'])
        return done

    def step(self,action):
        action = action.copy()
        if len(action.shape) > 1:
            action = action[0]
        steps_to_sim = int(np.round(self.nav_period / self.dynamics.h))
        self.lander.prev_state = self.lander.state.copy()
        BT, F, L, mdot = self.lander.thruster_model.thrust(action)
        for i in range(steps_to_sim): 
            self.dynamics.next(self.t, BT, F, L, mdot, self.lander)
            self.t += self.dynamics.h
            done = self.check_for_done(self.lander)
            if done:
                break
        
        self.steps+=1
        self.glideslope_constraint.calculate(self.lander.state)
        self.rh_constraint.step(self.lander.state)

        #######################
        # this is expensive, so only do it once per step
        #  1.) update the sensor state
        #  2.) check for sensor violation
        #  3.) get reward
        #  4.) update lander trajectory
        #
        agent_image, agent_state = self.lander.get_state_agent(self.t)

        if self.lander.sensor.check_for_vio():
            done = True
            if self.debug_done:
                print('FOV VIO')
            if self.steps <= 5 and self.debug_steps:
                print('FEW STEPS: ')
                print(self.lander.trajectory['position'])
                print(self.lander.trajectory['velocity'])
                print(self.lander.trajectory['thrust'])

        ########################

        reward,reward_info = self.reward_object.get( self.lander, action, done, self.steps, 
                                                    self.glideslope_constraint, self.attitude_constraint, 
                                                    self.w_constraint, self.rh_constraint)
        self.lander.update_trajectory(done, self.t)
        if done:
            self.episode += 1
        return agent_image , agent_state,reward,done,reward_info

    def test_policy_batch(self, agent , n, print_every=100, use_ts=False, keys=None, test_mode=True):
        t0 = time()
        if keys is None:
            keys1 = ['norm_vf', 'norm_rf', 'position', 'velocity', 'fuel', 'attitude_321', 'w', 'glideslope'] 
            keys2 =  ['thrust','glideslope']
            keys = keys1 + keys2

        all_keys = self.lander.get_engagement_keys()
        print('worked 1')
        agent.policy.test_mode = test_mode 
        self.lander.use_trajectory_list = True
        self.episode = 0
        self.lander.trajectory_list = []
        self.display_errors = True
        for i in range(n):
            agent.run_episode()
            for k in all_keys:
                if not k in keys:
                    self.lander.trajectory_list[-1][k] = None
            #self.test_policy_episode(policy, input_normalizer,use_ts=use_ts)
            if i % print_every == 0 and i > 0:
                print('i (et): %d  (%16.0f)' % (i,time()-t0 ) )
                t0 = time()
                self.lander.show_cum_stats()
                print(' ')
                self.lander.show_final_stats(type='final')
        print('')
        self.lander.show_cum_stats()
        print('')
        self.lander.show_final_stats(type='final')
        print('')
        self.lander.show_final_stats(type='ic')

    def test_policy_batch_notm(self, agent , n, print_every=100, use_ts=False):
        print('worked')
        self.lander.use_trajectory_list = True
        self.episode = 0
        self.lander.trajectory_list = []
        self.display_errors = True
        for i in range(n):
            agent.run_episode()
            #self.test_policy_episode(policy, input_normalizer,use_ts=use_ts)
            if i % print_every == 0 and i > 0:
                print('i : ',i)
                self.lander.show_cum_stats()
                print('')
                self.lander.show_final_stats(type='final')
        print('')
        self.lander.show_cum_stats()
        print('')
        self.lander.show_final_stats(type='final')
        print('')
        self.lander.show_final_stats(type='ic')


 
class RL_stats(object):
    
    def __init__(self,lander,logger,render_func,allow_plotting=True,print_every=1, vf=None, scaler=None):
        self.logger = logger
        self.render_func = render_func
        self.lander = lander
        self.scaler = scaler
        self.vf     = vf
        self.keys = ['r_f',  'v_f', 'r_i', 'v_i', 'norm_rf', 'norm_vf', 'gs_f', 'thrust', 'norm_thrust','fuel', 'rewards', 'fuel_rewards', 
                     'glideslope_rewards', 'glideslope_penalty', 'glideslope', 'norm_af', 'norm_wf', 'rh_penalty',
                     'att_rewards', 'att_penalty', 'attitude', 'w', 'a_f', 'w_f',
                     'w_rewards', 'w_penalty','fov_penalty', 'theta_cv', 'seeker_angles', 'cs_angles', 'optical_flow', 'v_err',
                     'landing_rewards','landing_margin', 'tracking_rewards', 'steps']
        self.formats = {}
        for k in self.keys:
            self.formats[k] = '{:8.2f}'
        self.formats['steps'] = '{:8.0f}'
        self.formats['optical_flow'] = '{:8.4f}'
        self.formats['cs_angles'] = '{:8.4f}'
        self.formats['v_err'] = '{:8.4f}'
        self.stats = {}
        self.history =  { 'Episode' : [] , 'MeanReward' : [], 'StdReward' : [] , 'MinReward' : [],  'Policy_KL' : [], 'Policy_Beta' : [], 'Variance' : [], 'Policy_Entropy' : [], 'ExplainedVarNew' :  [] , 
                          'Norm_rf' : [], 'Norm_vf' : [], 'SD_rf' : [], 'SD_vf' : [], 'Max_rf' : [], 'Max_vf' : [], 
                          'Model ExpVarOld' : [], 'Model P Loss Old' : [], 
                          'Norm_af' : [], 'Norm_wf' : [], 'SD_af' : [], 'SD_wf' : [], 'Max_af' : [], 'Max_wf' : [], 'MeanSteps' : [], 'MaxSteps' : []} 

        self.plot_learning = self.plot_agent_learning
        self.clear()
        
        self.allow_plotting = allow_plotting 
        self.last_time  = time() 

        self.update_cnt = 0
        self.episode = 0
        self.print_every = print_every 

        
        if allow_plotting:
            plt.clf()
            plt.cla()
            self.fig2 = plt.figure(2,figsize=plt.figaspect(0.5))
            self.fig3 = plt.figure(3,figsize=plt.figaspect(0.5))
            self.fig4 = plt.figure(4,figsize=plt.figaspect(0.5))
            self.fig5 = plt.figure(5,figsize=plt.figaspect(0.5))
            self.fig6 = plt.figure(6,figsize=plt.figaspect(0.5))
            self.fig7 = plt.figure(7,figsize=plt.figaspect(0.5))

    def save_history(self,fname):
        np.save(fname + "_history", self.history)

    def load_history(self,fname):
        self.history = np.load(fname + ".npy")

    def clear(self):
        for k in self.keys:
            self.stats[k] = []
    
    def update_episode(self,sum_rewards,steps):    
        self.stats['rewards'].append(sum_rewards)
        self.stats['fuel_rewards'].append(np.sum(self.lander.trajectory['fuel_reward']))
        self.stats['tracking_rewards'].append(np.sum(self.lander.trajectory['tracking_reward']))
        self.stats['glideslope_rewards'].append(np.sum(self.lander.trajectory['glideslope_reward']))
        self.stats['glideslope_penalty'].append(np.sum(self.lander.trajectory['glideslope_penalty']))
        self.stats['glideslope'].append(np.asarray(self.lander.trajectory['glideslope']))
        self.stats['att_penalty'].append(np.sum(self.lander.trajectory['att_penalty']))
        self.stats['rh_penalty'].append(np.sum(self.lander.trajectory['rh_penalty']))
        self.stats['fov_penalty'].append(np.sum(self.lander.trajectory['fov_penalty']))
        self.stats['seeker_angles'].append(self.lander.trajectory['seeker_angles'])
        self.stats['cs_angles'].append(self.lander.trajectory['cs_angles'])
        self.stats['optical_flow'].append(self.lander.trajectory['optical_flow'])
        self.stats['v_err'].append(self.lander.trajectory['v_err'])
        self.stats['theta_cv'].append(self.lander.trajectory['theta_cv'])
        self.stats['att_rewards'].append(np.sum(self.lander.trajectory['att_reward']))
        #self.stats['att_rewards'].append(np.asarray(self.lander.trajectory['tracking_reward']))
        
        self.stats['w_penalty'].append(np.sum(self.lander.trajectory['w_penalty']))
        self.stats['w_rewards'].append(np.sum(self.lander.trajectory['w_reward']))
        self.stats['landing_rewards'].append(np.sum(self.lander.trajectory['landing_reward'])) 
        self.stats['attitude'].append(self.lander.trajectory['attitude_321'])
        self.stats['w'].append(self.lander.trajectory['w'])

        self.stats['landing_margin'].append(np.sum(self.lander.trajectory['landing_margin']))
        self.stats['r_f'].append(self.lander.trajectory['position'][-1])
        self.stats['v_f'].append(self.lander.trajectory['velocity'][-1])
        self.stats['gs_f'].append(self.lander.trajectory['glideslope'][-1])
        self.stats['w_f'].append(self.lander.trajectory['w'][-1])
        self.stats['a_f'].append(self.lander.trajectory['attitude_321'][-1][1:3])
        self.stats['w_f'].append(self.lander.trajectory['w'][-1])
        self.stats['r_i'].append(self.lander.trajectory['position'][0])
        self.stats['v_i'].append(self.lander.trajectory['velocity'][0])
        self.stats['norm_rf'].append(self.lander.trajectory['norm_rf'][-1])
        self.stats['norm_vf'].append(self.lander.trajectory['norm_vf'][-1])
        self.stats['norm_af'].append(np.linalg.norm(self.lander.trajectory['attitude_321'][-1][1:3]))  # don't care about yaw
        self.stats['norm_wf'].append(np.linalg.norm(self.lander.trajectory['w'][-1]))

        self.stats['norm_thrust'].append(np.linalg.norm(self.lander.trajectory['thrust'],axis=1))
        self.stats['thrust'].append(self.lander.trajectory['thrust'])
        self.stats['fuel'].append(np.linalg.norm(self.lander.trajectory['fuel'][-1]))
        self.stats['steps'].append(steps)
        self.episode += 1

    def check_and_append(self,key):
        if key not in self.logger.log_entry.keys():
            val = 0.0
        else:
            val = self.logger.log_entry[key]
        self.history[key].append(val)
 
    # called by render at policy update 
    def show(self):
 
        self.history['MeanReward'].append(np.mean(self.stats['rewards']))
        self.history['StdReward'].append(np.std(self.stats['rewards']))
        self.history['MinReward'].append(np.min(self.stats['rewards']))

        self.check_and_append('Policy_KL')
        self.check_and_append('Policy_Beta')
        self.check_and_append('Variance')
        self.check_and_append('Policy_Entropy')
        self.check_and_append('ExplainedVarNew')
        self.check_and_append('Model ExpVarOld')
        self.check_and_append('Model P Loss Old')
        self.history['Episode'].append(self.episode)

        self.history['Norm_rf'].append(np.mean(self.stats['norm_rf']))
        self.history['SD_rf'].append(np.mean(self.stats['norm_rf']+np.std(self.stats['norm_rf'])))
        self.history['Max_rf'].append(np.max(self.stats['norm_rf']))

        self.history['Norm_vf'].append(np.mean(self.stats['norm_vf']))
        self.history['SD_vf'].append(np.mean(self.stats['norm_vf']+np.std(self.stats['norm_vf'])))
        self.history['Max_vf'].append(np.max(self.stats['norm_vf']))

        self.history['Norm_af'].append(np.mean(self.stats['norm_af']))
        self.history['SD_af'].append(np.mean(self.stats['norm_af']+np.std(self.stats['norm_af'])))
        self.history['Max_af'].append(np.max(self.stats['norm_af']))

        self.history['Norm_wf'].append(np.mean(self.stats['norm_wf']))
        self.history['SD_wf'].append(np.mean(self.stats['norm_wf']+np.std(self.stats['norm_wf'])))
        self.history['Max_wf'].append(np.max(self.stats['norm_wf']))


        self.history['MeanSteps'].append(np.mean(self.stats['steps']))
        self.history['MaxSteps'].append(np.max(self.stats['steps']))

        if self.allow_plotting:
            self.render_func(self.lander.trajectory,vf=self.vf,scaler=self.scaler)

            self.plot_rewards()
            self.plot_learning()
            self.plot_rf()
            self.plot_vf()
            self.plot_af()
            self.plot_wf()
        if self.update_cnt % self.print_every == 0:
            self.show_stats()
            self.clear()
        self.update_cnt += 1

    def show_stats(self):
        et = time() - self.last_time
        self.last_time = time()

        r_f = np.linalg.norm(self.stats['r_f'],axis=1)
        v_f = np.linalg.norm(self.stats['v_f'],axis=1)       
 
        f = '{:6.2f}'
        print('Update Cnt = %d    ET = %8.1f   Stats:  Mean, Std, Min, Max' % (self.update_cnt,et))
        for k in self.keys:
            f = self.formats[k]    
            v = self.stats[k]
            if k == 'glideslope' or k == 'thrust' or k=='tracking_error' or k=='norm_thrust' or k=='attitude' or k=='w' or k=='theta_cv' or k=='optical_flow' or k=='v_err' or k=='seeker_angles' or k=='cs_angles': 
                v = np.concatenate(v)
            v = np.asarray(v)
            if len(v.shape) == 1 :
                v = np.expand_dims(v,axis=1)
            s = '%-8s' % (k)
            #print('foo: ',k,v)
            s += envu.print_vector(' |',np.mean(v,axis=0),f)
            s += envu.print_vector(' |',np.std(v,axis=0),f)
            s += envu.print_vector(' |',np.min(v,axis=0),f)
            s += envu.print_vector(' |',np.max(v,axis=0),f)
            print(s)

        #print('R_F, Mean, SD, Min, Max: ',np.mean(r_f), np.std(r_f))
        #print('V_F, Mean, SD, Min, Max: ',np.mean(v_f), np.mean(v_f))
 
 
    def plot_rewards(self):
        self.fig2.clear()
        plt.figure(self.fig2.number)
        self.fig2.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()
        
        lns1=ax.plot(ep,self.history['MeanReward'],'r',label='Mean R')
        lns2=ax.plot(ep,np.asarray(self.history['MeanReward'])-np.asarray(self.history['StdReward']),'b',label='SD R')
        lns3=ax.plot(ep,self.history['MinReward'],'g',label='Min R')
        lns4=ax2.plot(ep,self.history['MaxSteps'],'c',linestyle=':',label='Max Steps')
        lns5=ax2.plot(ep,self.history['MeanSteps'],'m',linestyle=':',label='Mean Steps')

        lns = lns1+lns2+lns3+lns4+lns5
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Episode")

        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        ax = plt.gca()
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig2.canvas.draw()

    def plot_agent_learning(self):
        self.fig3.clear()
        plt.figure(self.fig3.number)
        self.fig3.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()
        lns1=ax.plot(ep,self.history['Policy_Entropy'],'r',label='Entropy')
        lns2=ax2.plot(ep,self.history['Policy_KL'],'b',label='KL Divergence')
        lns3=ax.plot(ep,self.history['ExplainedVarNew'],'g',label='Explained Variance')
        lns4=ax.plot(ep,self.history['Policy_Beta'],'k',label='Beta')
        foo = 10*np.asarray(self.history['Variance'])
        lns5=ax.plot(ep,foo,'m',label='10X Variance')


        lns = lns1+lns2+lns3+lns4+lns5
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Update")
        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig3.canvas.draw()

    def plot_model_learning(self):
        self.fig3.clear()
        plt.figure(self.fig3.number)
        self.fig3.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()
        lns1=ax.plot(ep,self.history['Model P Loss Old'],'r',label='Model Loss')
        lns2=ax2.plot(ep,self.history['Model ExpVarOld'],'b',label='Model ExpVar')

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Update")
        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig3.canvas.draw()

    def plot_rf(self):
        self.fig4.clear()
        plt.figure(self.fig4.number)
        self.fig4.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        
        plt.plot(ep,self.history['Norm_rf'],'r',label='Norm_rf (m)')
        plt.plot(ep,self.history['SD_rf'], 'b',linestyle=':',label='SD_rf (m)')
        plt.plot(ep,self.history['Max_rf'], 'g',linestyle=':',label='Max_rf (m)')
 
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig4.canvas.draw()

    def plot_vf(self):
        self.fig5.clear()
        plt.figure(self.fig5.number)
        self.fig5.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        
        plt.plot(ep,self.history['Norm_vf'],'r',label='Norm_vf (m/s)')
        plt.plot(ep,self.history['SD_vf'], 'b',linestyle=':',label='SD_vf (m/s)')
        plt.plot(ep,self.history['Max_vf'], 'g',linestyle=':',label='Max_vf (m/s)')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig5.canvas.draw()

    def plot_af(self):
        self.fig6.clear()
        plt.figure(self.fig6.number)
        self.fig6.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']

        plt.plot(ep,self.history['Norm_af'],'r',label='Norm_af')
        plt.plot(ep,self.history['SD_af'], 'b',linestyle=':',label='SD_af')
        plt.plot(ep,self.history['Max_af'], 'g',linestyle=':',label='Max_af')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig6.canvas.draw()

    def plot_wf(self):
        self.fig7.clear()
        plt.figure(self.fig7.number)
        self.fig7.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']

        plt.plot(ep,self.history['Norm_wf'],'r',label='Norm_wf')
        plt.plot(ep,self.history['SD_wf'], 'b',linestyle=':',label='SD_wf')
        plt.plot(ep,self.history['Max_wf'], 'g',linestyle=':',label='Max_wf')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig7.canvas.draw()
 
