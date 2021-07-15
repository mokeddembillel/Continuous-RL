import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity, dropout=False):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        
    return nn.Sequential(*layers)

class MLPQFunction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, activation,
            name='critic', chkpt_dir='tmp/sql'):
        super().__init__()
        self.action_dim = action_dim
        # self.state_layer = nn.Linear(state_dim,hidden_sizes[0])
        # self.action_layer = nn.Linear(action_dim,hidden_sizes[0])
        self.q = mlp([state_dim + action_dim] + hidden_dim + [1], activation, dropout=False)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print('device', self.device)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sql')

    def forward(self, obs, act,n_sample = 1):
        if n_sample > 1:
            obs = obs.unsqueeze(1).repeat(1,n_sample,1)
            assert obs.dim() == 3
        q = self.q(T.cat([act,obs], dim=-1))
        return  T.squeeze(q, -1).squeeze(-1)   # Critical to ensure q has right shape.
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class SamplingNetwork(nn.Module):
    def __init__(self,n_particles,batch_size,observation_space,action_space,hidden_sizes = (256,256),
                activation=nn.ReLU, name='actor', chkpt_dir='tmp/sql'):
        super().__init__()
        self.n_particles = n_particles
        self.batch_size = batch_size
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]

        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # self.state_layer = mlp([self.state_dim, hidden_sizes[0]], activation) 
        # self.noise_layer = mlp([self.action_dim, hidden_sizes[0]],activation)
        # self.layers = mlp(list(hidden_sizes) + [self.action_dim], activation, nn.Tanh)

        self.concat = mlp([self.state_dim + self.action_dim] + list(hidden_sizes),activation, dropout=False)
        self.layer2 =  mlp(list(hidden_sizes) + [self.action_dim], nn.Tanh, nn.Tanh, dropout=False)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sql')

    def _forward(self,state,n_particles = 1):
        n_state_samples = state.shape[0]

        if n_particles > 1:
            state = state.unsqueeze(1)
            state = state.repeat(1,n_particles,1)

            assert state.dim() == 3
            latent_shape = (n_state_samples, n_particles,
                            self.action_dim)
        else:
            latent_shape = (n_state_samples, self.action_dim)

        noise = T.rand(latent_shape).to(self.device) * 4-2
        # state_out = self.state_layer(state)
        # noise_out = self.noise_layer(noise)
        #print("noise_out.shape=",noise_out.shape,"state_out.shape=",state_out.shape)
        #tmp = state_out.unsqueeze(-1)
        #print("before tmp.shape=",tmp.shape)
        #tmp = tmp + noise_out
        #print("after tmp.shape=",tmp.shape)
        samples = self.concat(T.cat([state, noise],dim=-1))
        #print("samples.shape=",samples.shape)
        samples = self.layer2(samples)
        return T.tanh(samples) if n_state_samples > 1 else T.tanh(samples).squeeze(0)

    def forward(self,state,n_particles=1):
        return self._forward(state,n_particles=n_particles)

    def act(self,state,n_particles=1):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with T.no_grad():
            return self._forward(state,n_particles).cpu().detach().numpy()
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
