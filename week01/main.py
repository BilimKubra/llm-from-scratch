"""
Hafta 1 - Forward Pass'i elle hesaplama
Farkli agirlik/bias degerleriyle ciktinin nasil degistigini gozlemliyoruz.
"""
import math

def neuron_forward(inputs, weights, bias):
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    return weighted_sum + bias

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

if __name__ == "__main__":
    inputs = [1.0, 2.0, 3.0]

    denemeler = [
        {"weights": [0.2, -0.5, 0.1], "bias": 0.7},
        {"weights": [0.2, -0.5, 0.1], "bias": -2.0},   # bias dusurulunce ne oluyor?
        {"weights": [2.0, -5.0, 1.0], "bias": 0.7},    # agirliklar buyutulunce ne oluyor?
    ]

    for i, d in enumerate(denemeler, 1):
        z = neuron_forward(inputs, d["weights"], d["bias"])
        output = sigmoid(z)
        print(f"Deneme {i}: agirliklar={d['weights']}, bias={d['bias']} -> z={z:.3f}, sigmoid={output:.3f}")
