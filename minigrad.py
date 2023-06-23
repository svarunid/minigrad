import math

class Value:
    def __init__(self, data, children = ()) -> None:
        self.data = data
        self.grad = 0.0
        self.children = set(children)
        self._backwards = lambda: None

    def __add__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        data = self.data + other.data
        children = (self, other)
        out = Value(data, children)

        def _backwards():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backwards = _backwards

        return out
    
    def __mul__(self, other):
        other  = other if isinstance(other, Value) else Value(other)
        data = self.data * other.data
        children = (self, other)
        out = Value(data, children)

        def _backwards():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backwards = _backwards

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        data = self.data ** other
        children = (self, )
        out = Value(data, children)

        def _backwards():
            self.grad = other* (self.data ** (other-1)) * out.grad
        out._backwards = _backwards

        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    def exp(self):
        out = Value(math.exp(self.data), (self,))

        def _backward():
            self.grad = out.data * out.grad
        out._backwards = _backward
        
        return out
    
    def log(self):
        out = Value(-100 if self.data == 0 else math.log(self.data), (self,))

        def _backward():
            self.grad = ((self.data**-1) if self.data != 0 else 0) * out.grad
        out._backwards = _backward

        return out
    
    def backwards(self):
        sorted_nodes: list[Value] = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                sorted_nodes.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(sorted_nodes):
            node._backwards()