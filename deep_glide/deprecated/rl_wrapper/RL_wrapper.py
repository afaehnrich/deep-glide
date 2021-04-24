import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from .model import actors, critics#, Critic_lin_4x128, Actor_lin_4x128
from .utils import Memory, Memory_Img, NormalizedEnv
from deep_glide.deprecated.properties import properties
import threading
from queue import Queue
from gym import spaces

# nach: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

class DDPGagent:
    def __init__(self, action_space:spaces.Box, observation_space:spaces.Box, device, actor_type, critic_type, 
                actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99,
                tau=1e-2, max_memory_size=50000, load_from_disk=False):
        # Params
        self.device = device
        self.num_states = observation_space.shape[0]
        self.num_actions = action_space.shape[0]
        self.normalizer = NormalizedEnv(action_space)
        self.gamma = torch.tensor(gamma).to(device)
        self.tau = torch.tensor(tau).to(device)
        self.tauminus = torch.tensor(1 - tau).to(device)
        if load_from_disk:
            self.load_model(device)
        else:
            # Networks
            self.actor = actor_type(device, self.num_states, self.num_actions).to(device)
            self.actor_target = actor_type(device, self.num_states, self.num_actions).to(device)
            self.critic = critic_type(device, self.num_states + self.num_actions, self.num_actions).to(device)
            self.critic_target = critic_type(device, self.num_states + self.num_actions, self.num_actions).to(device)
        self.actor.share_memory()
        self.actor_target.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0).to(self.device))
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state)
            action = action.detach().cpu().numpy()[0]
        self.actor.train()
        return action
    
    def _update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        actions = self.normalizer.reverse_action(actions)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)
        # Actor loss
        self.critic.eval()
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.critic.train()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        # update
        self.actor_optimizer.zero_grad()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        policy_loss.backward()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self.actor_optimizer.step()     
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self.critic_optimizer.zero_grad()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        critic_loss.backward() 
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self.critic_optimizer.step()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (self.tauminus))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (self.tauminus))
       
    def update(self, batch_size):
        b_size = min(len(self.memory), batch_size)
        if b_size<2: return
        if len(self.memory) < batch_size: return
        self._update(b_size)

    def save_model(self):
        torch.save(self.actor, 'torch_models/actor.pt')
        torch.save(self.actor_target, 'torch_models/actor_target.pt')
        torch.save(self.critic, 'torch_models/critic.pt')
        torch.save(self.critic_target, 'torch_models/critic_target.pt')

    def load_model(self, device):
        self.actor = torch.load('torch_models/actor.pt').to(device)
        self.actor_target = torch.load('torch_models/actor_target.pt').to(device)
        self.critic = torch.load('torch_models/critic.pt').to(device)
        self.critic_target = torch.load('torch_models/critic_target.pt').to(device)


        
class DDPGagent_map(DDPGagent):
    def __init__(self, num_states, action_space, map_shape, device, actor_type, critic_type, 
                actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99,
                tau=1e-2, max_memory_size=50000, use_threading = False):
        # Params
        self.device = device
        self.num_states = num_states
        self.map_shape = map_shape
        self.num_actions = len(action_space)
        self.normalizer = NormalizedEnv(action_space)
        self.gamma = torch.tensor(gamma).to(device)
        self.tau = torch.tensor(tau).to(device)
        self.tauminus = torch.tensor(1 - tau).to(device)

        # Networks       
        self.actor = actor_type(device, self.num_states, self.map_shape, self.num_actions).to(device)
        self.actor.share_memory()
        self.actor_target = actor_type(device, self.num_states, self.map_shape, self.num_actions).to(device)
        self.actor_target.share_memory()
        self.critic = critic_type(device, self.num_states + self.num_actions, self.map_shape, self.num_actions).to(device)
        self.critic.share_memory()
        self.critic_target = critic_type(device, self.num_states + self.num_actions, self.map_shape, self.num_actions).to(device)
        self.critic_target.share_memory()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        # Training
        self.memory = Memory_Img(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        self.use_threading = use_threading
        if use_threading:
            self.q = Queue()
            self.thread = self.MyThread(self.q, args=(self._update,))
            self.thread.start()
    
    def get_action(self, state, img):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0).to(self.device))
        img = Variable(torch.from_numpy(img).float().unsqueeze(0).to(self.device))
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.forward(state, img)
            action = action.detach().cpu().numpy()[0]
        self.actor.train()
        return action
    
    def _update(self, batch_size):
        states, imgs, actions, rewards, next_states, next_imgs, _ = self.memory.sample(batch_size)
        actions = self.normalizer.reverse_action(actions)
        states = torch.FloatTensor(states).to(self.device)
        imgs = torch.FloatTensor(imgs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_imgs = torch.FloatTensor(next_imgs).to(self.device)
        # Critic loss
        Qvals = self.critic.forward(states, imgs, actions)
        next_actions = self.actor_target.forward(next_states, next_imgs)
        next_Q = self.critic_target.forward(next_states, next_imgs, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)
        # Actor loss
        self.critic.eval()
        policy_loss = -self.critic.forward(states, imgs, self.actor.forward(states, imgs)).mean()
        self.critic.train()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        # update
        self.actor_optimizer.zero_grad()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        policy_loss.backward()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self.actor_optimizer.step()     
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self.critic_optimizer.zero_grad()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        critic_loss.backward() 
        if self.device.type == 'cuda': torch.cuda.synchronize()
        self.critic_optimizer.step()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (self.tauminus))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (self.tauminus))
        
    def update(self, batch_size):
        b_size = min(len(self.memory), batch_size)
        if b_size<2: return
        #if len(self.memory) < batch_size: return
        self._update(b_size)

            