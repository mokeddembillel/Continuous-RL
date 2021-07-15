import gym
import numpy as np
from td3_torch import Agent
import pybulletgym
from gym import wrappers


#del gym.registry.env_specs['gym_lqr:lqr-v0']

#env = gym.make('gym_lqr:lqr-stochastic-v0')
#env = gym.make('gym_lqr:lqr-2d-v0')
#env = gym.make('gym_lqr:lqr-v0')
#env = gym.make('InvertedPendulumPyBulletEnv-v0')
#env = gym.make('InvertedPendulum-v2')
env = gym.make('Walker2DPyBulletEnv-v0')
#env = gym.make('Ant-v2')
#print(env.action_space.shape[0])


if __name__ == '__main__':
    #env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.001, beta=0.001,
            input_dims=env.observation_space.shape, tau=0.005,
            env=env, batch_size=100, layer1_size=400, layer2_size=300,
            n_actions=env.action_space.shape[0], action_bound=env.action_space.high)
    n_games = 3000
    filename = 'plots/' + 'LunarLanderContinuous_' + str(n_games) + '_games.png'

    best_score = env.reward_range[0]
    score_history = []
    steps_history = []
    #agent.load_models()

    for i in range(n_games):
        env.render()
        observation = env.reset()
        #observation = env.reset(init_x=np.array([100]), max_steps=200)
        done = False
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # print('state: ', np.squeeze(observation))
            # print('ac: ', np.squeeze(action))
            # print('re:', reward)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            steps += 1
            observation = observation_
            
        score_history.append(score)
        steps_history.append(steps)
        np.save('tmp/td3/score_history', np.array(score_history))
        np.save('tmp/td3/steps_history', np.array(steps_history))
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    #plot_learning_curve(x, score_history, filename)
