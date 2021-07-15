import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import gym
import numpy as np
from sac_torch import Agent
from gym import wrappers
import math
import torch as T
from networks import ActorNetwork, CriticNetwork
import pybulletgym




if __name__ == '__main__':

    
    #env = gym.make('InvertedPendulumPyBulletEnv-v0')
    #env = gym.make('gym_lqr:lqr-stochastic-v0')
    #env = gym.make('gym_lqr:lqr-v0')
    #env = gym.make('gym_lqr:lqr-2d-v0')
    #env = gym.make('InvertedPendulum-v2')
    env = gym.make('Walker2DPyBulletEnv-v0')
    #print(env.action_space.shape[0])
    actor = ActorNetwork(0.0003, input_dims=env.observation_space.shape, \
                         n_actions=env.action_space.shape[0], max_action=env.action_space.high)
        
    ActorNetwork.load_checkpoint(actor)
    env.render(mode='human')
    score_history = []
    steps_history = []
    
    for i in range(100):
        
        env.render()
        observation = env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            action = actor.sample_normal(T.Tensor([observation]).to(actor.device))
            action = action[0].cpu().detach().numpy().squeeze()
            #print(action)

            observation_, reward, done, info = env.step(action)
            
            score += reward
            steps += 1
            observation = observation_
            time.sleep(0.03)
            #print(done)
        score_history.append(score)
        steps_history.append(steps)
        avg_score = np.mean(score_history[-20:])

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
   
    env.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    