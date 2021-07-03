import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, n=30) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

ddpg = moving_average(np.load('ddpg/score_history.npy'))
td3 = moving_average(np.load('td3/score_history.npy'))
sac = moving_average(np.load('sac/score_history.npy'))
sql = moving_average(np.load('sql/score_history.npy'))


x = np.array(list(range(0, 1900)))

plt.figure()
plt.title('Cumulative reward for every episode')
axes = plt.gca()
axes.yaxis.grid()
plt.plot(x, ddpg[:1900], label='DDPG')
plt.plot(x, sac[:1900], color='red', label='SAC')
plt.plot(x, td3[:1900], color='green', label='TD3')
plt.plot(x, sql[:1900], color='orange', label='SQL')
plt.legend()
plt.savefig('score_history.png', dpi=200)

plt.show()



ddpg = moving_average(np.load('ddpg/steps_history.npy'))
td3 = moving_average(np.load('td3/steps_history.npy'))  
sac = moving_average(np.load('sac/steps_history.npy'))
sql = moving_average(np.load('sql/steps_history.npy'))


x = np.array(list(range(0, 1900)))

plt.figure()
plt.title('Number of steps in every episode')
axes = plt.gca()
axes.yaxis.grid()
plt.plot(x, ddpg[:1900], label='DDPG')
plt.plot(x, sac[:1900], color='red', label='SAC')
plt.plot(x, td3[:1900], color='green', label='TD3')
plt.plot(x, sql[:1900], color='orange', label='SQL')
plt.legend()
plt.savefig('steps_history.png', dpi=200)

plt.show()









