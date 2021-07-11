import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
import math
import torch as T
from networks import ActorNetwork, CriticNetwork
from multigoal import MultiGoalEnv


from plotter import QFPolicyPlotter

def plot(actor):
    paths = []
    actions_plot=[]
    env = MultiGoalEnv()
    n_games = 50
    max_episode_length = 30
    for i in range(n_games):
        observation = env.reset(init_state=[-3, 0])
        episode_length = 0
        done = False
        score = 0
        path = {'infos':{'pos':[]}}
        while not done:
            env.render()
            #print('state: ', np.squeeze(observation))
            action, _ = actor.forward(T.Tensor([observation]).to(actor.device))
            action = action.cpu().detach().numpy()[0]
            #print('ac: ', action[0].cpu().detach().numpy())
            observation_, reward, done, info = env.step(action)
            path['infos']['pos'].append(observation)
            
            if episode_length == max_episode_length:
                done = True
            episode_length += 1
            
            #print('re:', reward)
            score += reward
            observation = observation_
        paths.append(path)
        
        
    env.render_rollouts(paths, fout="test_%d.png" % i)
if __name__ == '__main__':

    
    #env = gym.make('InvertedPendulumPyBulletEnv-v0')
    #env = gym.make('gym_lqr:lqr-stochastic-v0')
    #env = gym.make('gym_lqr:lqr-v0')
    #env = gym.make('gym_lqr:lqr-2d-v0')
    #env = gym.make('InvertedPendulum-v2')
    env = MultiGoalEnv()
    #print(env.action_space.shape[0])
    actor = ActorNetwork(0.0003, input_dims=env.observation_space.shape, \
                         n_actions=env.action_space.shape[0], max_action=env.action_space.high)
        
    critic_1 = CriticNetwork(0.0003, input_dims=env.observation_space.shape, \
                    n_actions=env.action_space.shape[0], name='critic_1')
    critic_2 = CriticNetwork(0.0003, input_dims=env.observation_space.shape, \
                    n_actions=env.action_space.shape[0], name='critic_2')
        
    ActorNetwork.load_checkpoint(actor)
    critic_1.load_checkpoint()
    critic_2.load_checkpoint()
    
    
    plot(actor)
    
    # plotter = QFPolicyPlotter(qf = critic_1, policy=actor, obs_lst=[[0,0]], default_action =[np.nan,np.nan], n_samples=100)
    # plotter.draw()
    
    
    


