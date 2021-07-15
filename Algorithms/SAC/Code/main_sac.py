# pybullet_envs
import gym
import numpy as np
import pybulletgym
from sac_torch import Agent
from gym import wrappers
import math
#import gym_lqr


if __name__ == '__main__':
    #env = gym.make('gym_lqr:lqr-stochastic-v0')
    #env = gym.make('gym_lqr:lqr-2d-v0')
    #env = gym.make('gym_lqr:lqr-v0')
    #env = gym.make('InvertedPendulumPyBulletEnv-v0')
    #env = gym.make('InvertedPendulum-v2')
    env = gym.make('Walker2DPyBulletEnv-v0')
    #env = gym.make('Ant-v2')
    #print(env.action_space.shape[0])
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 3000 
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'inverted_pendulum.png'
    
    #print(env.action_space.high)
    
    figure_file = 'plots/' + filename
    
    best_score = env.reward_range[0]
    score_history = []
    steps_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        
        #observation = env.reset(init_x=np.array([7., 7., 5.]), max_steps=200)
        #observation = env.reset(init_x=np.array([100, 100]), max_steps=200)
        #init_x = np.array([np.random.choice(np.array([-100, 100]))])
        # init_x = np.random.uniform(low=-100, high=100, size=(1,))
        # observation = env.reset(init_x=init_x, max_steps=15)
        env.render()
        observation = env.reset()

        #print(observation)
        done = False
        score = 0
        steps = 0
        while not done:
            
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # print('state: ', np.squeeze(observation))
            # print('ac: ', np.squeeze(action))
            # print('re:', reward)
            #print('Q: ', np.squeeze(env.get_Q()))
            score += reward
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        steps_history.append(steps)
        np.save('tmp/sac/score_history', np.array(score_history))
        np.save('tmp/sac/steps_history', np.array(steps_history))
        avg_score = np.mean(score_history[-20:])
        
        if avg_score > best_score:
            best_score = avg_score
            # if not load_checkpoint:
            #     agent.save_models()
            #     print('P: ', env.get_P())
            #     np.save('tmp/sac/optimal_P', env.get_P())
                
        
        #print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'init state %.1f' % np.squeeze(init_x))
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
    env.close()        
    # if not load_checkpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)


# import gym
# env = gym.make('InvertedPendulum-v2')
# env.reset()
# for _ in range(1000):
#     env.render()
#     a = env.action_space.sample()
#     env.step(a) # take a random action
#     print(a)
# env.close()





