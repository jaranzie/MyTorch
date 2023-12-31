import jax.numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        se     = np.square(A-Y)
        sse    = np.sum(se)
        mse    = sse/(N*C)
        
        return mse
    
    def backward(self):
    
        dLdA = (self.A-self.Y) 
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")

        self.softmax     = np.exp(A) / (np.exp(A) @ Ones_C @ Ones_C.T) # TODO
        crossentropy     = -Y * np.log(self.softmax)
        sum_crossentropy = np.sum(crossentropy)
        # sum_crossentropy = Ones_N.T @ crossentropy @ Ones_C
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y
        
        return dLdA
