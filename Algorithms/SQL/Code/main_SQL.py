# pybullet_envs
import gym
import numpy as np
import pybulletgym
from SQL_torch import Agent
from gym import wrappers
import math
import gym_lqr
import torch as T

if __name__ == '__main__':
    
    
    #env = gym.make('gym_lqr:lqr-v0')
    env = gym.make('gym_lqr:lqr-2d-v0')
    #env = gym.make('gym_lqr:lqr-stochastic-v0')
    
    #env = gym.make('InvertedPendulum-v2')

    print(env.observation_space.shape)
    print(env.action_space.shape)
    
    agent = Agent(env, hidden_dim=[128, 128], replay_size=int(1e6), pi_lr=3e-4, 
                  q_lr=3e-4, batch_size=128, n_particles=16, gamma=0.99, polyak=0.995)
    
    #print(env.action_space.high)
    
    
    n_games = 100
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename
    best_score = env.reward_range[0]
    score_history = []
    state_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        #observation = env.reset(init_x=np.array([10., 10., 10.]), max_steps=200)
        observation = env.reset(init_x=np.array([5, 5]), max_steps=100)
        #observation = env.reset(init_x=np.array([50]), max_steps=100)
        #observation = env.reset()
        done = False
        score = 0
        state = 0
        n_particles = 16
        while not done:
            env.render()
            
            # Sample an action for st using f
            action = agent.get_sample(observation, n_sample=n_particles)
            Q_values = agent.Q_Network(T.tensor(observation).unsqueeze(0).float().to(agent.Q_Network.device), T.from_numpy(action).float().unsqueeze(0).to(agent.Q_Network.device),n_sample=n_particles)
            mm_weights = agent.compute_millowmax_target(Q_values).float().detach().numpy().squeeze()
            ind = np.random.choice(np.array([i for i in range(0, n_particles)]), p=mm_weights)
            action = action[ind]
            
            print('st', observation.squeeze())
            print('ac: ', action)
            # Sample next state from the environment
            observation_, reward, done, info = env.step(action)
            # Sum the rewards
            score += reward
            #state += observation
            print('re:', reward)

            # Save the new experience in the replay memor
            agent.replay_buffer.store(observation, action, reward, observation_, done)
            
            #if not load_checkpoint:
            if not load_checkpoint:
                batch = agent.replay_buffer.sample_batch(agent.batch_size)
                #print("before updating..")
                agent.learn(0, data=batch)  
            
            observation = observation_
        
        # Unimportant for now
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])
        
        #state_history.append(state / 100)
#        avg_state = np.mean(state_history[-100:])

        
        # if avg_score > best_score:
        #     best_score = avg_score
        #     if not load_checkpoint:
        #         agent.save_models()
                
        #print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'avg_state %.1f' % avg_state)
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
    env.close()        
    # if not load_checkpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)








