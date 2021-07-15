import gym
import numpy as np
import pybulletgym
from SQL_torch import Agent
from gym import wrappers
import math
from multigoal import MultiGoalEnv
import torch as T
from plotter import QFPolicyPlotter

from copy import deepcopy

if __name__ == "__main__":
    
    env= MultiGoalEnv()
    
    agent = Agent(env, hidden_dim=[256,256], replay_size=int(1e6), 
        pi_lr=1e-3, q_lr=1e-3, batch_size=100, n_particles=16, gamma=0.99, polyak=0.995)
    
    epochs=100
    update_after=0
    max_ep_len=30
    steps_per_epoch=400
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    
    
    
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    n_particles=16
    epsilon = 0.8
    for t in range(total_steps):
        
        if epsilon > 0.2:
            epsilon -= 0.00001
        
        
        #a = agent.get_sample(o)
        if np.random.uniform(0,1) > epsilon: 
            a = agent.get_sample(o, n_sample=n_particles)
            # ind = np.random.choice(np.array([i for i in range(0, n_particles)]))
            # a = a[ind]
            Q_values = agent.Q_Network(T.tensor(o).float().unsqueeze(0).to(agent.Q_Network.device), T.from_numpy(a).float().unsqueeze(0).to(agent.Q_Network.device),n_sample=n_particles)
            mm_weights = agent.compute_millowmax_target(Q_values).float().detach().numpy().squeeze()
            ind = np.random.choice(np.array([i for i in range(0, n_particles)]), p=mm_weights)
            a = a[ind]
        else:
            a = np.random.uniform(-1,1, size=(2))
        
        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        #print("t=",t,"ep_ret=",ep_ret, "ep_len=",ep_len)

        d = False if ep_len==max_ep_len else d

        agent.replay_buffer.store(o, a, r, o2, d)

        
        o = o2

        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

            
        
        # Update handling
        if t >= update_after:
            batch = agent.replay_buffer.sample_batch(agent.batch_size)
            #print("before updating..")
            agent.learn(t, data=batch)
            if t % 500 == 0:
                agent.target_Q_Network = deepcopy(agent.Q_Network)
            #print("after updating..")

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            print("epoch=",epoch)
            agent.plot_paths(epoch)
            plotter = QFPolicyPlotter(qf = agent.Q_Network, policy=agent.SVGD_Network, obs_lst=[[0,0],[-2.5,-2.5],[2.5,2.5]], default_action =[np.nan,np.nan], n_samples=100)
            plotter.draw()

            # Save model
            #if (epoch % save_freq == 0) or (epoch == epochs):
                #logger.save_state({'env': env}, None)

