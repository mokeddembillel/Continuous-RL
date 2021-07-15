import gym
import numpy as np
import pybulletgym
from SQL_torch import Agent
from gym import wrappers
import math
from multigoal import MultiGoalEnv
import torch as T
from plotter import QFPolicyPlotter
from networks import SamplingNetwork

from copy import deepcopy

# import gym
# env = gym.make('Walker2d-v2')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()


if __name__ == "__main__":
    
    #env= MultiGoalEnv()
    env = gym.make('Walker2DPyBulletEnv-v0')
    print(env.observation_space.high)
    print(env.action_space.high)
    agent = Agent(env, hidden_dim=[128, 128], replay_size=int(1e6), pi_lr=3e-4, 
                  q_lr=3e-4, batch_size=128, n_particles=32, gamma=0.99, polyak=0.995)
    
    reward_scale = 10
    epochs=5000
    update_after=1
    steps_per_epoch=1000
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    
    
    
    best_score = 0.0
    # Main loop: collect experience in env and update/log each epoch
    score_history = []
    steps_history = []
    n_particles=32
    count = 0
    for t in range(epochs):
        #env.render()
        o= env.reset()
        ep_ret, ep_steps = 0, 0
        d = False
        epsilon = 0.8
        target_updated = False
        while not d:
            #a = agent.get_sample(o, n_sample=n_particles)
            if epsilon > 0.1:
                epsilon -= 0.0001
        
            #a = agent.get_sample(o)

            # a = agent.get_sample(o)
            if np.random.uniform(0,1) > epsilon: 
                a = agent.get_sample(o, n_sample=n_particles)
                # ind = np.random.choice(np.array([i for i in range(0, n_particles)]))
                # a = a[ind]
                Q_values = agent.Q_Network(T.tensor(o).float().unsqueeze(0).to(agent.Q_Network.device), T.from_numpy(a).float().unsqueeze(0).to(agent.Q_Network.device),n_sample=n_particles)
                # mm_weights = agent.compute_millowmax_target(Q_values).float().detach().numpy().squeeze()
                # ind = np.random.choice(np.array([i for i in range(0, n_particles)]), p=mm_weights)
                ind = T.argmax(Q_values)
                a = a[ind]
            else:
                a = np.random.uniform(-1,1, size=(env.action_space.shape[0]))
            
            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            
            # Store experience to replay buffer
            agent.replay_buffer.store(o, a, r, o2, d)
    
            # most recent observation!
            o = o2
            # Update handling
            if 2 >= update_after:
                batch = agent.replay_buffer.sample_batch(agent.batch_size)
                #print("before updating..")
                agent.learn(t, data=batch)
                if t % 10 == 0 and not target_updated:
                    target_updated = True
                    agent.target_Q_Network = deepcopy(agent.Q_Network)
                #print("after updating..")
    
            ep_steps += 1
            
            
        score_history.append(ep_ret)
        steps_history.append(ep_steps)
        np.save('tmp/sql/score_history', np.array(score_history))
        np.save('tmp/sql/steps_history', np.array(steps_history))
        avg_score = np.mean(score_history[-10:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            
        
        
        
        
        print("episode=",t,"ep_ret=",ep_ret, "ep_len=",ep_steps)
        
        


    # plotter = QFPolicyPlotter(qf = agent.Q_Network, ss=agent.SVGD_Network, obs_lst=[[0,0],[-2.5,-2.5],[2.5,2.5],[-2.5,2.5],[2.5,-2.5]], default_action =[np.nan,np.nan], n_samples=100)
    # plotter.draw()
    


