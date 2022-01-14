import torch
from torch import nn

class ExtendedResidualNeuralNetwork(nn.Module):
    
    def __init__(self, input_size, depth, expansion, activation=nn.Tanh()):
        super(ExtendedResidualNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.depth = depth
        
        self.expansion = expansion
        self.width = self.expansion * self.input_size
        
        self.linear_layers = nn.ModuleList(nn.Linear(self.width, self.width) for _ in range(depth-1))

        self.linear_collapse = nn.Linear(self.width, self.input_size)

        self.activation = activation

        return
    
        
    def forward(self, x_in):

        exp_dims = [1 for d in x_in.shape]
        exp_dims[-1] = self.expansion
        
        z = x_in.repeat(*exp_dims)
        
        for linear in self.linear_layers:
            z = z + self.activation(linear(z))

        y = self.linear_collapse(z)

        return y
    
    