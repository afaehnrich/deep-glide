import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable
import logging
#nach https://gist.github.com/cyoon1729


class Critic_lin_4x128_map(nn.Module):
    def __init__(self, device, input_size, map_shape, output_size):
        n=128
        f = int(input_size/output_size)
        logging.debug('4f={} 2f={} f={}'.format(4*f, 32*f, f))
        super(Critic_lin_4x128, self).__init__()
        self.batchNorm = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, max(f*4,n))
        self.linear2 = nn.Linear(max(f*4,n), max(f*2,n))
        self.linear3 = nn.Linear(max(f*2,n), max(f,n))
        self.linear4 = nn.Linear(max(f,n), output_size)
        self.device = device
                
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        #state = state.to(self.device)
        #action = action.to(self.device)
        x = torch.cat([state, action], 1)#.to(self.device)
        x = self.batchNorm(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class Actor_lin_4x128_map(nn.Module):
    def __init__(self, device, input_size, map_shape, output_size, learning_rate = 3e-4):
        n=32
        f = int(input_size/output_size)
        super(Actor_lin_4x128, self).__init__()
        self.batchNorm = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, max(f*4,n))
        self.linear2 = nn.Linear(max(f*4,n), max(f*2,n))
        self.linear3 = nn.Linear(max(f*2,n), max(f,n))
        self.linear4 = nn.Linear(max(f,n), output_size)
        self.device = device
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        #state = state.to(self.device)
        x = state
        x = self.batchNorm(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        x = torch.tanh(x) #Only use this with NormalizedEnv
        return x

class Critic_lin_4x128(nn.Module):
    def __init__(self, device, input_size, output_size):
        super(Critic_lin_4x128, self).__init__()
        n=128
        f = int(input_size/output_size)
        logging.debug('4f={} 2f={} f={}'.format(4*f, 2*f, f))
        self.batchNorm = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, max(f*4,n))
        self.linear2 = nn.Linear(max(f*4,n), max(f*2,n))
        self.linear3 = nn.Linear(max(f*2,n), max(f,n))
        self.linear4 = nn.Linear(max(f,n), output_size)
        self.device = device
                
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        #state = state.to(self.device)
        #action = action.to(self.device)
        x = torch.cat([state, action], 1)#.to(self.device)
        x = self.batchNorm(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class Actor_lin_4x128(nn.Module):
    def __init__(self, device, input_size, output_size, learning_rate = 3e-4):
        super(Actor_lin_4x128, self).__init__()
        n=128
        f = int(input_size/output_size)
        self.batchNorm = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, max(f*4,n))
        self.linear2 = nn.Linear(max(f*4,n), max(f*2,n))
        self.linear3 = nn.Linear(max(f*2,n), max(f,n))
        self.linear4 = nn.Linear(max(f,n), output_size)
        self.device = device
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        #state = state.to(self.device)
        x = state
        x = self.batchNorm(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        x = torch.tanh(x) #Only use this with NormalizedEnv
        return x
        
class Critic_lin_4x1024(nn.Module):
    def __init__(self, device, input_size, output_size):
        super(Critic_lin_4x1024, self).__init__()
        self.batchNorm = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, output_size)
        self.device = device
                
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        #state = state.to(self.device)
        #action = action.to(self.device)
        x = torch.cat([state, action], 1)#.to(self.device)
        x = self.batchNorm(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x

class Actor_lin_4x1024(nn.Module):
    def __init__(self, device, input_size, output_size, learning_rate = 3e-4):
        super(Actor_lin_4x1024, self).__init__()
        self.batchNorm = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, output_size)
        self.device = device
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = state
        x = self.batchNorm(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x =self.linear4(x)
        x = torch.tanh(x) #Only use this with NormalizedEnv
        return x
        
class Critic_lin_10x1024(nn.Module):
    def __init__(self, device, input_size, output_size):
        super(Critic_lin_10x1024, self).__init__()
        self.batchNorm = nn.BatchNorm1d(input_size)
        self.linearIn = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 1024)
        self.linear5 = nn.Linear(1024, 1024)
        self.linear6 = nn.Linear(1024, 1024)
        self.linear7 = nn.Linear(1024, 1024)
        self.linear8 = nn.Linear(1024, 1024)
        self.linear9 = nn.Linear(1024, 1024)
        self.linearOut = nn.Linear(1024, output_size)
        self.device = device
                
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        #state = state.to(self.device)
        #action = action.to(self.device)
        x = torch.cat([state, action], 1)#.to(self.device)
        x = self.batchNorm(x)
        x = F.relu(self.linearIn(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = F.relu(self.linear7(x))
        x = F.relu(self.linear8(x))
        x = F.relu(self.linear9(x))
        x = self.linearOut(x)

        return x

class Actor_lin_10x1024(nn.Module):
    def __init__(self, device, input_size, output_size, learning_rate = 3e-4):
        super(Actor_lin_10x1024, self).__init__()
        self.batchNorm = nn.BatchNorm1d(input_size)
        self.linearIn = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.linear4 = nn.Linear(1024, 1024)
        self.linear5 = nn.Linear(1024, 1024)
        self.linear6 = nn.Linear(1024, 1024)
        self.linear7 = nn.Linear(1024, 1024)
        self.linear8 = nn.Linear(1024, 1024)
        self.linear9 = nn.Linear(1024, 1024)
        self.linearOut = nn.Linear(1024, output_size)
        self.device = device
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = state
        x = self.batchNorm(x)
        x = F.relu(self.linearIn(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = F.relu(self.linear7(x))
        x = F.relu(self.linear8(x))
        x = F.relu(self.linear9(x))
        x =self.linearOut(x)
        x = torch.tanh(x) #Only use this with NormalizedEnv
        return x

class critics:
    lin_4x128 = Critic_lin_4x128
    lin_4x1024 = Critic_lin_4x1024
    lin_10x1024 = Critic_lin_10x1024

class actors:
    lin_4x128 = Actor_lin_4x128
    lin_4x1024 = Actor_lin_4x1024
    lin_10x1024 = Actor_lin_10x1024
