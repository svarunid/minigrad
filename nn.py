import random
from minigrad import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    
class Neuron(Module):
    def __init__(self, n_input) -> None:
        self.weights = [Value(random.uniform(1, -1)) for _ in range(n_input)]
        self.bias = Value(random.uniform(1, -1))

    def __call__(self, x):
        out =  sum((xi * wi for xi, wi in zip(x, self.weights)), self.bias)
        return out
    
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer(Module):
    def __init__(self, n_input, n_output) -> None:
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class Sequential(Module):
    def __init__(self, *args: Layer):
        self.layers = args
        
    def __call__(self, x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
class ReLU(Module):
    def __call__(self, x):
        out_list = []
        for i in x:
            out = Value(0 if i.data < 0 else i.data, (i,))
            def _backward():
                i.grad += (out.data > 0) * out.grad
            i._backwards = _backward
            out_list.append(i)
        return out_list
    
    def parameters(self):
        return []
    
class Sigmoid(Module):
    def __call__(self, x):
        out_list = []
        for i in x:
            out = i.exp() / (i.exp() + 1)
            out_list.append(out)
        return out_list