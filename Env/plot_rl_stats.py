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

class RL_stats(object):
    
    def __init__(self,lander,logger,history, allow_plotting=True,print_every=1, vf=None, scaler=None):
        self.logger = logger
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
        self.history =   history 

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
            self.fig2 = plt.figure(20,figsize=plt.figaspect(0.5))
            self.fig3 = plt.figure(30,figsize=plt.figaspect(0.5))
            self.fig4 = plt.figure(40,figsize=plt.figaspect(0.5))
            self.fig5 = plt.figure(50,figsize=plt.figaspect(0.5))
            self.fig6 = plt.figure(60,figsize=plt.figaspect(0.5))
            self.fig7 = plt.figure(70,figsize=plt.figaspect(0.5))

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

        self.check_and_append('KL')
        self.check_and_append('Beta')
        self.check_and_append('Variance')
        self.check_and_append('PolicyEntropy')
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
        
        lns1=ax.plot(ep,self.history['MeanReward'],'r',label='Mean')
        lns2=ax.plot(ep,np.asarray(self.history['MeanReward'])-np.asarray(self.history['StdReward']),'b',label='Mean - StdDev')
        lns3=ax.plot(ep,self.history['MinReward'],'g',label='Min')

        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rewards over Rollout Batch")
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
        lns1=ax.plot(ep,self.history['PolicyEntropy'],'r',label='Entropy')
        lns2=ax2.plot(ep,self.history['KL'],'b',label='KL Divergence')
        lns3=ax.plot(ep,self.history['ExplainedVarNew'],'g',label='Explained Variance')
        lns4=ax.plot(ep,self.history['Beta'],'k',label='Beta')
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
        
        plt.plot(ep,self.history['Norm_rf'],'r',label='Mean')
        plt.plot(ep,self.history['SD_rf'], 'b',linestyle=':',label='Mean + StdDev')
        plt.plot(ep,self.history['Max_rf'], 'g',linestyle=':',label='Max ')
 
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Norm Miss (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig4.canvas.draw()

    def plot_vf(self):
        self.fig5.clear()
        plt.figure(self.fig5.number)
        self.fig5.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        
        plt.plot(ep,self.history['Norm_vf'],'r',label='Mean')
        plt.plot(ep,self.history['SD_vf'], 'b',linestyle=':',label='Mean + StdDev')
        plt.plot(ep,self.history['Max_vf'], 'g',linestyle=':',label='Max')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Terminal Speed  (m/s)") 
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
 
