# pybullet_envs
import gym
import numpy as np
import pybulletgym
from sac_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
import math
from multigoal import MultiGoalEnv
import torch as T
from plotter import QFPolicyPlotter

def plot(agent):
    paths = []
    actions_plot=[]
    env = MultiGoalEnv()
    n_games = 50
    max_episode_length = 20
    for i in range(n_games):
        observation = env.reset(init_state=[0, 0])
        episode_length = 0
        done = False
        score = 0
        path = {'infos':{'pos':[]}}
        while not done:
            env.render()
            #print('state: ', np.squeeze(observation))
            action = agent.choose_action(observation)
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
        
        score = score / 200
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])
        
    env.render_rollouts(paths, fout="test_%d.png" % i)


if __name__ == '__main__':
    env = MultiGoalEnv()
    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 500
    
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    #print(env.action_space.high)
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    
    max_episode_length = 30

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        
        observation = env.reset(init_state=[0, 0])
        episode_length = 0
        
        done = False
        score = 0
        while not done:
            env.render()
            #print('state: ', np.squeeze(observation))
            action = agent.choose_action(observation)
            #print('ac: ', np.squeeze(action))
            observation_, reward, done, info = env.step(action)
            
            if episode_length == max_episode_length:
                done = True
            episode_length += 1
            
            #print('re:', reward)
            #print('Q: ', np.squeeze(env.get_Q()))
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            
        plotter = QFPolicyPlotter(qf = agent.critic_1, policy=agent.actor, obs_lst=[[-2,0],[0,2],[2.5,2.5]], default_action =[np.nan,np.nan], n_samples=50)
        plotter.draw() 
        
        score = score
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])
        
        if avg_score > best_score:
            best_score = avg_score
            # if not load_checkpoint:
            #     agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
        
        plot(agent)
        
    env.close()  
    
    

    
    


    

