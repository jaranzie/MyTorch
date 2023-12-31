import jax.numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:
    
    def forward(self, Z):
        # We have a vector Z, which we need to run a sigmoid function on each element of the vector.

        self.A = 1 / (1 + np.exp(-Z))
        
        return self.A
    
    def backward(self):
    
        dAdZ = self.A * (1 - self.A)
        
        return dAdZ


class Tanh:
    
    def forward(self, Z):
    
        self.A = np.sinh(Z) / np.cosh(Z)
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - self.A**2
        
        return dAdZ


class ReLU:
    
    def forward(self, Z):
    
        self.A = np.maximum(np.zeros_like(Z), Z)

        return self.A
    
    def backward(self):
    
        dAdZ = np.where(self.A > 0, np.ones_like(self.A), np.zeros_like(self.A))
        
        return dAdZ
        
        
