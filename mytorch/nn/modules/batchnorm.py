import jax.numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        
        if eval:

            return NotImplemented
            
        self.Z         = Z
        self.N         = Z.shape[0]
        
        self.M         = np.sum(self.Z, axis=0) / self.N # 1 x C
        self.V         = np.sum((self.Z - self.M) ** 2, axis=0) / self.N # 1 x C
        self.NZ        = (Z - self.M) / (np.sqrt(self.V + self.eps))
        self.BZ        = self.BW * self.NZ + self.Bb
        
        self.running_M = self.running_M * self.alpha + self.M
        self.running_V = self.running_V * self.alpha + self.V
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBW  = np.sum(dLdBZ * self.NZ, axis=0)
        self.dLdBb  = np.sum(dLdBZ, axis=0)
        
        dLdNZ       = dLdBZ * self.W # TODO
        dLdV        = np.sum(dLdNZ * (self.Z - self.M) * (-0.5 * (self.V + self.eps) ** (-1.5)), axis=0)
        dLdM        = None # TODO
        
        dLdZ        = None # TODO
        
        return  NotImplemented