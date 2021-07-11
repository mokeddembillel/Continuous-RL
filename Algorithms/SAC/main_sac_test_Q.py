import gym
import numpy as np

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
    states = np.expand_dims(np.expand_dims(np.arange(-100, 100, 5, dtype=np.float32), -1), -1)
    actions = np.expand_dims(np.expand_dims(np.arange(-10, 10, 0.5, dtype=np.float32), -1), -1)
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
    plt
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("States")
    ax.set_ylabel("Actions")
    ax.set_zlabel("Q Values")
    
    # Plot a basic wireframe.
    #ax.plot_wireframe(X, Y, q_env, rstride=10, cstride=10)
    ax.plot_wireframe(X, Y, q_network, rstride=10, cstride=10, color='red', label='Learned Q')
    ax = plt.gca()
    ax.plot_wireframe(X, Y, q_env, rstride=10, cstride=10, color='blue', label='Analytical Q')

    plt.show()
    
    
    
    # #plt.plot(states, q_network[0, :])
    # #plt.plot(actions, q_network[0, :])
    # #plt.plot(states, q_env[:, 0])
    # plt.plot(states, q_env[:, 0], color='blue')
    # plt.plot(states, q_env[:, 5], color='green')
    # plt.plot(states, q_env[:, 10], color='orange')
    # plt.plot(states, q_env[:, 15], color='red')
    # plt.show()

##################################################################################



# Create States and Actions
    states = np.expand_dims(np.expand_dims(np.arange(-100, 105, 5, dtype=np.float32), -1), -1)
    actions = np.expand_dims(np.expand_dims(np.arange(-10, 10.5, 0.5, dtype=np.float32), -1), -1)
    # states = []
    # actions = []
    
    #print(states)
    states = np.expand_dims(np.expand_dims(np.arange(-100, 105, 5, dtype=np.float32), -1), -1)
    actions = np.expand_dims(np.expand_dims(np.arange(-10, 10.5, 0.5, dtype=np.float32), -1), -1)
    
    observation = env.reset(init_x=np.array([100]), max_steps=2000)
    for i, x in enumerate(states):
    #for i in range(-100, 100, 10):
        q_env = []
        q_network = []
        for u in actions:
            q_env.append(np.squeeze(env.get_Q(x,u)))
            q1_new_policy = np.squeeze(critic_1.forward(T.tensor(x), T.tensor(u)).detach().numpy())
            q2_new_policy = np.squeeze(critic_2.forward(T.tensor(x), T.tensor(u)).detach().numpy())
            q = np.minimum(q1_new_policy, q2_new_policy)
            q_network.append(q)
        
        q_env = q_env / np.max(np.abs(q_env))
        q_network = q_network / np.max(np.abs(q_network))
        
        states = np.squeeze(states)
        actions = np.squeeze(actions)
        q_env = np.squeeze(q_env)
        q_network = np.squeeze(q_network)
        
        
        
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        
        strr = ' '.join(['The Q values for a range of actions where the state is', str(x[0, 0])])
        plt.plot(actions, q_network, color='red', label='Learned Q')
        plt.plot(actions, q_env, color='blue', label='Analytical Q')
        plt.title(strr, fontsize=8)
        axes = plt.gca()
        axes.yaxis.grid()
        plt.legend()
        name = '_'.join(['fig', str(i)])
        plt.savefig(name, dpi=200)
        plt.show()
        
        states = np.expand_dims(np.expand_dims(np.arange(-100, 105, 5, dtype=np.float32), -1), -1)
        actions = np.expand_dims(np.expand_dims(np.arange(-10, 10.5, 0.5, dtype=np.float32), -1), -1)
    

####################################################################################




# # Create States and Actions
#     states = np.expand_dims(np.expand_dims(np.arange(-100, 100, 5, dtype=np.float32), -1), -1)
#     actions = np.expand_dims(np.expand_dims(np.arange(-10, 10, 0.5, dtype=np.float32), -1), -1)
#     # states = []
#     # actions = []
    
#     observation = env.reset(init_x=np.array([100]), max_steps=2000)
    
#     x = np.array([[-60]], dtype=np.float32)
#     q_env = []
#     q_network = []
#     for u in actions:
#         q_env.append(np.squeeze(env.get_Q(x, u)))
#         q1_new_policy = np.squeeze(critic_1.forward(T.tensor(x), T.tensor(u)).detach().numpy())
#         q2_new_policy = np.squeeze(critic_2.forward(T.tensor(x), T.tensor(u)).detach().numpy())
#         q = np.minimum(q1_new_policy, q2_new_policy)
#         q_network.append(q)
    
#     q_env = q_env / np.max(np.abs(q_env))
#     q_network = q_network / np.max(np.abs(q_network))
    
#     states = np.squeeze(states)
#     actions = np.squeeze(actions)
#     q_env = np.squeeze(q_env)
#     q_network = np.squeeze(q_network)
    
    
    
#     from mpl_toolkits.mplot3d import axes3d
#     import matplotlib.pyplot as plt
    
#     strr = ' '.join(['state', str(x[0, 0]), ' action', str(u[0, 0])])
#     plt.plot(actions, q_env, color='green')
#     plt.plot(actions, q_network, color='red')
#     plt.title(strr)
#     plt.show()
    


####################################################################################
# Create States and Actions
    # states = np.expand_dims(np.expand_dims(np.arange(-100, 100, 5, dtype=np.float32), -1), -1)
    
    # observation = env.reset(init_x=np.array([100]), max_steps=2000)
    # q_env = []
    # q_network = []
    # for x in states:
    #     q_env.append(np.squeeze(env.get_V(x)))
        
    
    # q_env = q_env / np.max(np.abs(q_env))
    
    # states = np.squeeze(states)
    # q_env = np.squeeze(q_env)
    # q_network = np.squeeze(q_network)
    
    
    
    # from mpl_toolkits.mplot3d import axes3d
    # import matplotlib.pyplot as plt
    
    # strr = ' '.join(['state', str(x[0, 0])])
    # plt.plot(states, q_env, color='green')
    # plt.title(strr)
    # plt.show()
    
    