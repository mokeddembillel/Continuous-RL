import gym
import numpy as np
import pybulletgym
from sac_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
import math
import torch as T
from networks import ActorNetwork, CriticNetwork
import gym_lqr
if __name__ == '__main__':

    
    #env = gym.make('InvertedPendulumPyBulletEnv-v0')
    #env = gym.make('gym_lqr:lqr-stochastic-v0')
    env = gym.make('gym_lqr:lqr-v0')
    #env = gym.make('InvertedPendulum-v2')

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
    
    # Load optimal P
    env.set_P(np.load('tmp/sac/optimal_P.npy'))
    
    
    # Create States and Actions
    states = np.expand_dims(np.expand_dims(np.arange(-10, 10, 0.5, dtype=np.float32), -1), -1)
    actions = np.expand_dims(np.expand_dims(np.arange(-1, 1, 0.05, dtype=np.float32), -1), -1)
    # states = []
    # actions = []
    q_env = []
    q_network = []
    #print(states)
    
    observation = env.reset(init_x=np.array([100]), max_steps=2000)

    
    for x in states:
        q_env.append([])
        q_network.append([])
        for u in actions:
            # states.append(x)
            # actions.append(u)
            q_env[-1].append(np.squeeze(env.get_Q(x, u)))
            #q_env.append(np.squeeze(env.get_Q(x, u)))
            q1_new_policy = np.squeeze(critic_1.forward(T.tensor(x), T.tensor(u)).detach().numpy())
            q2_new_policy = np.squeeze(critic_2.forward(T.tensor(x), T.tensor(u)).detach().numpy())
            #print(q1_new_policy)
            q = np.minimum(q1_new_policy, q2_new_policy)
            #q_network.append(q)
            q_network[-1].append(q)
            
    q_env = q_env / np.max(np.abs(q_env))
    q_network = q_network / np.max(np.abs(q_network))
    
    states = np.squeeze(states)
    actions = np.squeeze(actions)
    q_env = np.squeeze(q_env)
    q_network = np.squeeze(q_network)
    
    X, Y = np.meshgrid(states, actions)

    
    # print(states.shape)
    # print(actions.shape)
    # print(q_env.shape)
    
    
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("States")
    ax.set_ylabel("Actions")
    ax.set_zlabel("Q Values")
    
    # Plot a basic wireframe.
    #ax.plot_wireframe(X, Y, q_env, rstride=10, cstride=10)
    ax.plot_wireframe(X, Y, q_network, rstride=10, cstride=10, color='red')
    ax = plt.gca()
    ax.plot_wireframe(X, Y, q_env, rstride=10, cstride=10, color='blue')

    plt.show()
    
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, projection='3d')
    
    
    # # Plot a basic wireframe.
    # ax2.plot_wireframe(X, Y, q_env, rstride=10, cstride=10)
    # #ax.plot_wireframe(X, Y, q_network, rstride=10, cstride=10)
    
    # plt.show()
    
    