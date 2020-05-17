
#variational autoencoder class

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        #encode
        self.conv1 = nn.Conv2d(1,5,kernel_size=3) #first convolution layer
        self.conv2 = nn.Conv2d(5,10,kernel_size=3) #second convolution layer
        self.fc1mu = nn.Linear(40,20) #mu layer
        self.fc1lv = nn.Linear(40,20) #logvariance layer
        
        #decode
        self.fc2 = nn.Linear(20, 75)
        self.fc3 = nn.Linear(75, 196)
        
    #return mean and variance of encoded data
    def encode(self, x):
        #perform convolution
        x = func.relu(func.max_pool2d(self.conv1(x), 2)) #perform convolution and pool
        x = func.relu(func.max_pool2d(self.conv2(x) , 2)) #convolve and pool
        x = x.view(-1,40) #resize for linear layers
        return self.fc1mu(x), self.fc1lv(x) #return mu and logmax
    
    #draw random sample
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.empty(mu.size()).normal_()
            return eps*mu + std
        else:
            return mu
        
    def decode(self, z):
        z = func.relu(self.fc2(z))
        return func.sigmoid(self.fc3(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    #loss function
    def loss_function(self, x_reconstructed, x, mu, logvar):
        #calculate binary cross entropy
        BCE = func.binary_cross_entropy(x_reconstructed, x.view(-1, 14*14))
    
        #calculate KL distance
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        norm = mu.size()[0]
        KLD /= norm*14*14
        
        return BCE + KLD
    
    #train model
    def backprop(self, x, optimizer):
        self.train()
        optimizer.zero_grad()
        x_reconstructed, mu, logvar = self(x)
        
        loss = self.loss_function(x_reconstructed, x, mu, logvar)
        loss.backward()
        
        optimizer.step()
        
        return loss
    
    #test model
    def test(self, x):
        self.eval()
        x_reconstructed, mu, logvar = self(x)
        loss = self.loss_function(x_reconstructed, x, mu, logvar)
        return loss
    
    
    