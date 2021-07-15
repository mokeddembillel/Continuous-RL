import numpy as np
import torch

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer():
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
#         self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
#         self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
#         self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


# class ReplayBuffer():
#     def __init__(self, max_size, input_shape, n_actions):
#         self.mem_size = max_size
#         self.mem_cntr = 0
#         self.state_memory = np.zeros((self.mem_size, input_shape))
#         self.new_state_memory = np.zeros((self.mem_size, input_shape))
#         self.action_memory = np.zeros((self.mem_size, n_actions))
#         self.reward_memory = np.zeros(self.mem_size)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

#     def store_transition(self, state, action, reward, state_, done):
#         index = self.mem_cntr % self.mem_size

#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward
#         self.terminal_memory[index] = done

#         self.mem_cntr += 1

#     def sample_buffer(self, batch_size):
#         max_mem = min(self.mem_cntr, self.mem_size)

#         batch = np.random.choice(max_mem, batch_size)

#         states = self.state_memory[batch]
#         states_ = self.new_state_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         dones = self.terminal_memory[batch]

#         return states, actions, rewards, states_, dones


