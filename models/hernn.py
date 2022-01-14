import torch
from torch import nn

class HamiltonianExtendedResidualNeuralNetwork(nn.Module):
    
    def __init__(self, d, depth, expansion, activation=nn.Tanh()):
        super(HamiltonianExtendedResidualNeuralNetwork, self).__init__()
        
        self.d = d # Input dimension is 2*d
        self.depth = depth
        
        self.expansion = expansion

        self.width = self.expansion * 2*self.d
        
        self.activation = activation
        # It is important that we use a differentiable activation function for the dynamical
        # backpropagation to work properly. Also, activation functions with gradient zero over
        # large parts of the domain are ill suited, as this constrains the outcome of the 
        # dynamical backpropagation considerably. As the dynamical backpropagation should be
        # a continuous function, activation functions that are note C^1 should be avoided.
        # These considerations imply that ReLU is a poor choice here.
        
        self.linear_layers = nn.ModuleList(nn.Linear(self.width, self.width) for _ in range(depth-1))
        self.linear_to_hamiltonian = nn.Linear(self.width, 1)

        
        self.zeroer = torch.optim.SGD(self.parameters(), lr=1e0) # Used to zero the gradients of
                                                                 # the parameters after the 
                                                                 # dynamical backpropagation.  

        return
    
        
    def forward(self, x_in):
        
        x = x_in.detach()
        x.requires_grad = True

        exp_dims = [1 for d in x_in.shape]
        exp_dims[-1] = self.expansion
        
        z = x.repeat(*exp_dims)
        
        for linear in self.linear_layers:
            z = z + self.activation(linear(z))
            
        H = self.linear_to_hamiltonian(z)
        
        H.sum().backward(retain_graph=True, create_graph=True)
        
        self.zeroer.zero_grad()

        y = torch.column_stack((x.grad[:,self.d:], -x.grad[:,:self.d]))

        x.grad.zero_()
        x.requires_grad = False
        
        return y
    
    
    def hamiltonian(self, x):
        
        exp_dims = [1 for d in x.shape]
        exp_dims[-1] = self.expansion

        z = x.repeat(*exp_dims)
   
        for linear in self.linear_layers:
            z = z + self.activation(linear(z))
            
        H = self.linear_to_hamiltonian(z)
        
        return H
