from ddpg_torch import Agent
import gym
import numpy as np
#from utils import plotLearning
import pybulletgym
from gym import wrappers


#del gym.registry.env_specs['gym_lqr:lqr-v0']
import gym_lqr

#env = gym.make('gym_lqr:lqr-stochastic-v0')
#env = gym.make('gym_lqr:lqr-2d-v0')
#env = gym.make('gym_lqr:lqr-v0')
#env = gym.make('InvertedPendulumPyBulletEnv-v0')
#env = gym.make('InvertedPendulum-v2')
env = gym.make('Walker2DPyBulletEnv-v0')
#env = gym.make('Ant-v2')
#print(env.action_space.shape[0])
print(env.action_space.high)
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0], 
              action_bound=env.action_space.high)

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
    observation = env.reset()

    #print(observation)
    done = False
    score = 0
    steps = 0
    while not done:
        #env.render()
        #print('state: ', np.squeeze(observation))
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        #print('ac: ', np.squeeze(action))
        #print('re:', reward)
        #print('Q: ', np.squeeze(env.get_Q()))
        score += reward
        steps += 1
        agent.remember(observation, action, reward, observation_, done)
        if not load_checkpoint:
            agent.learn()
        observation = observation_
    score_history.append(score)
    steps_history.append(steps)
    #np.save('tmp/ddpg/score_history', np.array(score_history))
    #np.save('tmp/ddpg/steps_history', np.array(steps_history))
    avg_score = np.mean(score_history[-20:])
    
    if avg_score > best_score:
        best_score = avg_score
        #if not load_checkpoint:
            #agent.save_models()
            #print('P: ', env.get_P())
            #np.save('tmp/sac/optimal_P', env.get_P())
            
    
    print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
env.close()        
# if not load_checkpoint:
#     x = [i+1 for i in range(n_games)]
#     plot_learning_curve(x, score_history, figure_file)
