"""
Hafta 1 - Forward Pass'i elle hesaplama
Hicbir kutuphane kullanmadan (NumPy bile yok) tek bir "neuron"un
girisleri nasil ciktiya cevirdigini gormek icin.
"""

def neuron_forward(inputs, weights, bias):
    # Her giris * agirlik carpimlarinin toplami + bias
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    z = weighted_sum + bias
    return z

def sigmoid(z):
    import math
    return 1 / (1 + math.exp(-z))

if __name__ == "__main__":
    inputs = [1.0, 2.0, 3.0]
    weights = [0.2, -0.5, 0.1]
    bias = 0.7

    z = neuron_forward(inputs, weights, bias)
    output = sigmoid(z)

    print(f"Agirlikli toplam (z): {z}")
    print(f"Aktivasyon sonrasi cikti: {output}")
