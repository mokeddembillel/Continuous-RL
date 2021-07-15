import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import gym
import numpy as np
from td3_torch import Agent
from gym import wrappers
import math
import torch as T
from networks import ActorNetwork, CriticNetwork
import pybulletgym

from torch.distributions.normal import Normal



if __name__ == '__main__':

    
    #env = gym.make('InvertedPendulumPyBulletEnv-v0')
    #env = gym.make('gym_lqr:lqr-stochastic-v0')
    #env = gym.make('gym_lqr:lqr-v0')
    #env = gym.make('gym_lqr:lqr-2d-v0')
    #env = gym.make('InvertedPendulum-v2')
    env = gym.make('Walker2DPyBulletEnv-v0')
    #print(env.action_space.shape[0])
    
    actor = ActorNetwork(0.001, env.observation_space.shape, 400, 300,
            n_actions=env.action_space.shape[0], name='actor')    
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
            state = T.tensor(observation, dtype=T.float).to(actor.device)
            mu = actor.forward(state).to(actor.device)
            mu_prime = mu + T.tensor(np.random.normal(scale=0.1),
                                        dtype=T.float).to(actor.device)
    
            mu_prime = T.clamp(mu_prime, env.action_space.low[0], env.action_space.high[0])
    
            action = (mu_prime  * T.tensor(env.action_space.high)).cpu().detach().numpy()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    