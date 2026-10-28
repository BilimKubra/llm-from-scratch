"""
Hafta 3 - PyTorch tensor temelleri.
Hafta 2'de Value sinifiyla elle yaptigimiz seyleri simdi PyTorch'un
gercek tensor API'siyle yapiyoruz.
"""
import torch

# --- Tensor olusturma ---
x = torch.tensor([1.0, 2.0, 3.0])
w = torch.tensor([0.2, -0.5, 0.1], requires_grad=True)
b = torch.tensor(0.7, requires_grad=True)

print("x:", x)
print("w:", w)
print("x.shape:", x.shape)

# --- Broadcasting: farkli boyutlardaki tensorler nasil birlesir ---
# Skaler bir sayi, her elemana otomatik "yayilir" (broadcast)
print("\nBroadcasting ornegi:")
print("x + 10 =", x + 10)  # 10, [10,10,10] gibi davranir

# --- Matrix multiplication (Hafta 1'deki forward pass'in tensor hali) ---
# Hafta 1'de: sum(xi * wi) + b  -- burada: torch.dot(x, w) + b
z = torch.dot(x, w) + b
print("\nz (agirlikli toplam + bias):", z)

output = torch.tanh(z)
print("output (tanh sonrasi):", output)

# --- Autograd: PyTorch kendi gradyanini hesapliyor ---
output.backward()
print("\nw.grad:", w.grad)
print("b.grad:", b.grad)
