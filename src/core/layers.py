# src/core.layers.py
# Linear, Conv2D, Pool, Softmax 등

import numpy as np
from .tensor import Tensor

class Linear:
    '''
    Linear Layer : y = [x,1] @ [w;b]
    '''

    def __init__(self, in_dim, out_dim):
        # Xavier init
        limit = np.sqrt(( 6 / (in_dim + out_dim))) # Xavier init
        self.W = Tensor(-limit, limit,
                        (in_dim+1, out_dim), 
                        requires_grad=True, 
                        name="W")
    
    def forward(self, x:Tensor):
        ones = Tensor(np.ones((x.data.shape[0], 1))) # bias용 column
        self.x = Tensor(np.concatenate([x.data, ones.data], axis=1), requires_grad=x.requires_grad)

        return self.x @ self.W
    
    def backward(self, grad_output: Tensor):
            """
            grad_output: dL/dY
            목표:
            - self.W_ext.grad  (W와 b 둘 다 포함)
            - dL/dX 반환
            """

            # dL/dW_ext  = X_ext^T @ dL/dY
            self.W_ext.grad = self.x_ext.data.T @ grad_output.data

            # dL/dX = dL/dY @ W_ext[:-1].T  (마지막 row는 bias라 제외)
            W = self.W_ext.data[:-1, :]            # bias 제외 part
            grad_input = grad_output.data @ W.T

            return Tensor(grad_input)


class ReLU:
    def forward(self, x: Tensor):
        self.x = x
        return x.relu()

    def backward(self, grad_output: Tensor):
        return Tensor(grad_output.data * (self.x.data > 0))


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: Tensor):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "W_ext"):
                params.append(layer.W_ext)
        return params