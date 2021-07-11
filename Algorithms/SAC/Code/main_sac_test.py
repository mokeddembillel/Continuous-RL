import gym
import numpy as np
import pybulletgym
from sac_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
import math
import torch as T
from networks import ActorNetwork
import gym_lqr
if __name__ == '__main__':

    
    #env = gym.make('InvertedPendulumPyBulletEnv-v0')
    env = gym.make('gym_lqr:lqr-stochastic-v0')
    env = gym.make('gym_lqr:lqr-v0')
    #env = gym.make('InvertedPendulum-v2')

    print(env.action_space.shape[0])
    actor = ActorNetwork(0.0003, input_dims=env.observation_space.shape, \
                         n_actions=env.action_space.shape[0], max_action=env.action_space.high)
    ActorNetwork.load_checkpoint(actor)
    
    #observation = env.reset(np.array([50, 50, 50]), 200)
    observation = env.reset(init_x=np.array([100]), max_steps=2000)

    print(observation)
    for _ in range(2000):
        env.render()
        state = T.Tensor([observation]).to(actor.device)
        action, _ = actor.sample_normal(state, reparameterize=False)
        observation, reward, _, _= env.step(action.cpu().detach().numpy()[0]*2) # take a random action

        print('state: ', np.squeeze(observation))
        # print('ac: ', np.squeeze(action))
        # print('re:', np.squeeze(reward))
        #print('Q: ', np.squeeze(env.get_Q()))
        
    env.close()
    
    
    """ env = gym.make('InvertedPendulum-v2')
    
    observation = env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        print(action)
        env.step(action) # take a random action
    env.close()"""
    
    
    