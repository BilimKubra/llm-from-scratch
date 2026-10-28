"""
Hafta 3 - Matrix multiplication ile batch (toplu) islem.
Hafta 1-2'de tek bir ornegi isliyorduk. Gercek egitimde binlerce ornegi
AYNI ANDA islemek icin matrix multiplication kullanilir - bu da GPU'larin
neden bu kadar hizli oldugunun temel sebebi.
"""
import torch

# 4 ornek (batch), her biri 3 ozellikli (Hafta 1'deki tek ornek yerine simdi 4 tane)
X = torch.tensor([
    [1.0, 2.0, 3.0],
    [0.5, -1.0, 2.0],
    [-1.0, 1.0, 0.0],
    [2.0, 2.0, 2.0],
])  # shape: (4, 3) -> 4 ornek, 3 ozellik

W = torch.tensor([0.2, -0.5, 0.1], requires_grad=True)  # shape: (3,)
b = torch.tensor(0.7, requires_grad=True)

print("X.shape:", X.shape)
print("W.shape:", W.shape)

# Tek bir ornek icin: dot(x, w) + b
# 4 ornek icin AYNI ANDA: X @ W + b  (matrix-vector carpimi)
z = X @ W + b
print("\nz (4 ornek icin ayni anda hesaplandi):", z)

output = torch.tanh(z)
print("output:", output)

loss = output.sum()
loss.backward()
print("\nW.grad (4 ornegin HEPSInden gelen gradyanlarin toplami):", W.grad)
