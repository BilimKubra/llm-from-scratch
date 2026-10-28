"""
Value sinifini kullanarak Neuron, Layer ve MLP (Multi-Layer Perceptron).
Karpathy'nin micrograd'indaki nn.py'a dayanarak.
"""
import random
from week02.engine import Value


class Neuron:
    def __init__(self, nin):
        # Her girdi icin rastgele bir agirlik, bir de bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Hafta 1'deki forward_pass ile birebir ayni mantik,
        # sadece artik Value kullandigimiz icin gradyan otomatik takip ediliyor
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        # nouts ornegin [4, 4, 1] ise: nin -> 4 -> 4 -> 1 katmanlari
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
