import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
#from networks import ActionValueNetwork, SamplerNetwork
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from networks import MLPQFunction, SamplingNetwork
import torch.optim as optim
import time
from copy import deepcopy


from multigoal import MultiGoalEnv

class Agent():
    def __init__(self, env_fn, hidden_dim, replay_size, gamma, pi_lr, q_lr, batch_size, n_particles, polyak):

        self.env= env_fn
      
        self.gamma = gamma
        self.n_particles = n_particles
        self.batch_size = batch_size
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.polyak = polyak
                
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        
        # Create actor-critic module and target networks
        self.Q_Network = MLPQFunction(self.state_dim, self.action_dim, hidden_dim, T.nn.ReLU)
        self.target_Q_Network = deepcopy(self.Q_Network)
        
        self.SVGD_Network = SamplingNetwork(batch_size = self.batch_size, n_particles = self.n_particles,
                    observation_space= self.env.observation_space, action_space = self.env.action_space)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.target_Q_Network.parameters():
            p.requires_grad = False
        
        # Set up optimizers for policy and q-function
        self.q_optimizer = optim.Adam(self.Q_Network.parameters(), lr=q_lr)
        self.SVGD_Network_optimizer = optim.Adam(self.SVGD_Network.parameters(), lr=pi_lr)
        
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.state_dim, act_dim=self.action_dim, size=replay_size)

    def save_models(self):
        print('.... saving models ....')
        self.Q_Network.save_checkpoint()
        self.SVGD_Network.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.Q_Network.load_checkpoint()
        self.SVGD_Network.load_checkpoint()
        

    def rbf_kernel(self, input_1, input_2,  h_min=1e-3):
        k_fix, out_dim1 = input_1.size()[-2:]
        k_upd, out_dim2 = input_2.size()[-2:]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        dist_sq = diff.pow(2).sum(-1)
        dist_sq = dist_sq.unsqueeze(-1)
        
        # Get median.
        median_sq = T.median(dist_sq, dim=1)[0]
        median_sq = median_sq.unsqueeze(1)
        
        h = median_sq / np.log(k_fix + 1.) + .001
        
        kappa = T.exp(-dist_sq / h)
        
        # Construct the gradient
        kappa_grad = -2. * diff / h * kappa
        return kappa, kappa_grad
    
    
    def compute_millowmax_target(self, Q_values):
        beta = 1
        mm_weights = []
        
        max_Q = T.max(Q_values, dim=1)[0]
        #print(Q_values)
        #mm_target = T.logsumexp(Q_values, dim=1) + T.log(T.tensor(1/self.n_particles))
        
        denominator = T.sum(T.exp(beta * Q_values - max_Q.unsqueeze(-1)), dim=1)
        #denominator = T.sum(T.exp(beta * Q_values), dim=1)
        #print(denominator)
        
        for i in range(self.n_particles):
            current_Q = Q_values[:, i]
            current_Q_exp = T.exp(beta * current_Q - max_Q)
            #current_Q_exp = T.exp(beta * current_Q)
            mm_weights.append(list((current_Q_exp / denominator).cpu().detach().numpy()))
            
        mm_weights = T.tensor(np.array(mm_weights, dtype=np.double)).view(-1, self.n_particles)
        #print(mm_weights)
        #print(mm_weights.sum())
            
        return mm_weights
    
    
    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    
        aplus = T.from_numpy(self.SVGD_Network.act(o2.to(self.SVGD_Network.device),n_particles=self.n_particles))
        #print("aplus=",aplus.shape)
        
        q = self.Q_Network(o.to(self.Q_Network.device),a.to(self.Q_Network.device)).to(self.Q_Network.device)
        
        # Bellman backup for Q function
        with T.no_grad():
            Q_soft_ = self.target_Q_Network(o2.to(self.target_Q_Network.device), aplus.to(self.target_Q_Network.device),n_sample=self.n_particles)
            
            V_soft_ = T.logsumexp(Q_soft_, dim=1).to(self.target_Q_Network.device)
            V_soft_ += self.action_dim * T.log(T.tensor([2.])).to(self.target_Q_Network.device)
            backup = r.to(self.target_Q_Network.device) + self.gamma * (1 - d.to(self.target_Q_Network.device)) * V_soft_
           
            # mm_weights = self.compute_millowmax_target(Q_soft_).to(self.target_Q_Network.device)
            # mm_target = T.sum(mm_weights * Q_soft_, dim=1)
            #mm_target = T.sum(mm_weights * (Q_soft_ - T.log(mm_weights)), dim=1)
            # backup = r.to(self.target_Q_Network.device) + self.gamma * (1 - d.to(self.target_Q_Network.device)) * mm_target.to(self.target_Q_Network.device)
            
            # Q_soft_ = T.max(self.target_Q_Network(o2, aplus,n_sample=self.n_particles), dim=1)[0]
            # backup = r + self.gamma * (1 - d) * Q_soft_
    
        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()
    
        # Useful info for logging
        loss_info = dict(QVals=q.cpu().detach().numpy())
    
       
        return loss_q, loss_info

    
    
    def update_svgd_ss(self, data,train=True):
        o = data['obs']
        actions = self.SVGD_Network(o.to(self.SVGD_Network.device),n_particles=self.n_particles)
        assert actions.shape == (self.batch_size,self.n_particles, self.action_dim)
        
        fixed_actions = self.SVGD_Network.act(o.to(self.SVGD_Network.device),n_particles=self.n_particles)
        fixed_actions = T.from_numpy(fixed_actions).to(self.SVGD_Network.device)
        fixed_actions.requires_grad = True
        svgd_target_values = self.Q_Network(o.to(self.Q_Network.device), fixed_actions.to(self.Q_Network.device),n_sample = self.n_particles)
    
        log_p = svgd_target_values
    
    
        grad_log_p = T.autograd.grad(log_p.sum().to(self.SVGD_Network.device), fixed_actions.to(self.SVGD_Network.device))[0]
        grad_log_p = grad_log_p.view(self.batch_size,self.n_particles, self.action_dim).unsqueeze(2).to(self.SVGD_Network.device)
        grad_log_p = grad_log_p.detach()
        assert grad_log_p.shape == (self.batch_size, self.n_particles, 1, self.action_dim)
    
        kappa, gradient = self.rbf_kernel(input_1=fixed_actions, input_2=actions)
    
        # Kernel function in Equation 13:
        # kappa = kappa.unsqueeze(dim=3)
        assert kappa.shape == (self.batch_size, self.n_particles, self.n_particles, 1)
    
        # Stein Variational Gradient in Equation 13:
        # T_C = total_steps/steps_per_epoch
        # anneal = ((t%(T_C))/T_C)**2
        anneal = 1.
        action_gradients = (1/self.n_particles)*T.sum(anneal*kappa * grad_log_p + gradient, dim=1).to(self.SVGD_Network.device)
        assert action_gradients.shape == (self.batch_size, self.n_particles, self.action_dim)
    
        # Propagate the gradient through the policy network (Equation 14).
        if train:
            self.SVGD_Network_optimizer.zero_grad()
            T.autograd.backward(-actions,grad_tensors=action_gradients)
            #T.nn.utils.clip_grad_norm_(g_net.parameters(),2)
            self.SVGD_Network_optimizer.step()
    
    def learn(self, t, data):
        #print("done update_svgd_ss")
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        
        # Finally, update target networks by polyak averaging.
        # with T.no_grad():
        #     for p, p_targ in zip(self.Q_Network.parameters(), self.target_Q_Network.parameters()):
        #         # NB: We use an in-place operations "mul_", "add_" to update target
        #         # params, as opposed to "mul" and "add", which would make new tensors.
        #         p_targ.data.mul_(self.polyak)
        #         p_targ.data.add_((1 - self.polyak) * p.data)
                
        # update stein sampler 
        if t%1 == 0:
          self.update_svgd_ss(data)
    

     
    def get_sample(self, o,n_sample=1):
        #self.SVGD_Network.eval()
        a = self.SVGD_Network.act(T.as_tensor(o, dtype=T.float32).to(self.SVGD_Network.device),n_particles=n_sample) * self.action_bound
        #self.SVGD_Network.train()        
        return np.clip(a, -self.action_bound, self.action_bound) 
    
    
    def plot_paths(self, epoch):
        paths = []
        actions_plot=[]
        env = MultiGoalEnv()
    
        for episode in range(50):
            observation = env.reset()
            done = False
            step = 0
            path = {'infos':{'pos':[]}}
            particles = None
            while not done and step < 30 :
               
                self.SVGD_Network.eval()
                actions = self.get_sample(observation,1)
                self.SVGD_Network.train()
               
                observation, reward, done, _ = env.step(actions)
                path['infos']['pos'].append(observation)
                step +=1
                paths.append(path)
        print("saving figure..., epoch=",epoch)
           
        env.render_rollouts(paths,fout="test_%d.png" % epoch)
    