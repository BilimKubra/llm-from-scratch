"""
Hafta 7 - Manuel backpropagation.
Hafta 5'teki embedding+MLP modelinin ileri gecisini ayni tutuyoruz, ama
geri gecisi (backward) ELLE, PyTorch'un autograd'ini KULLANMADAN hesapliyoruz.
Sonra PyTorch'un gercek autograd gradyanlariyla karsilastiriyoruz.
"""
import torch
import torch.nn.functional as F

words = ["ada", "ali", "aslan", "kaya", "deniz", "elif", "can", "ece", "mert", "zeynep"]
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
vocab_size = len(stoi)
BLOCK_SIZE = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

X, Y = build_dataset(words)
n = X.shape[0]
EMB_DIM = 4
HIDDEN_DIM = 16

g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, EMB_DIM), generator=g, requires_grad=True)
W1 = torch.randn((BLOCK_SIZE * EMB_DIM, HIDDEN_DIM), generator=g, requires_grad=True)
b1 = torch.randn(HIDDEN_DIM, generator=g, requires_grad=True)
W2 = torch.randn((HIDDEN_DIM, vocab_size), generator=g, requires_grad=True)
b2 = torch.randn(vocab_size, generator=g, requires_grad=True)

# --- Ileri gecis (PyTorch'un kendi islemlerini kullaniyoruz, ama her ara
#     degeri saklayip retain_grad() ile autograd'in gradyanini da gorecegiz) ---
emb = C[X]
embflat = emb.view(n, -1)
preact = embflat @ W1 + b1
h = torch.tanh(preact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)

for t in [emb, embflat, preact, h, logits]:
    t.retain_grad()
for p in [C, W1, b1, W2, b2]:
    p.grad = None

loss.backward()  # gercek autograd gradyanlari - karsilastirma icin referans

# --- Simdi AYNI gradyanlari ELLE, zincir kuralini kullanarak hesaplayalim ---

# 1) cross_entropy + softmax'in birlesik turevi: softmax(logits) - one_hot(Y), / n
dlogits = F.softmax(logits, dim=1)
dlogits[range(n), Y] -= 1
dlogits /= n

# 2) logits = h @ W2 + b2  ->  d(logits)/dh = W2, d(logits)/dW2 = h, d(logits)/db2 = 1
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)

# 3) h = tanh(preact)  ->  d(tanh)/dx = 1 - tanh(x)^2
dpreact = (1.0 - h ** 2) * dh

# 4) preact = embflat @ W1 + b1
dembflat = dpreact @ W1.T
dW1 = embflat.T @ dpreact
db1 = dpreact.sum(0)

# 5) embflat = emb.view(n, -1)  ->  sadece sekil degisikligi, gradyan da aynen geri sekillenir
demb = dembflat.view(emb.shape)

# 6) emb = C[X]  ->  her context'teki her karakterin embedding'i C'nin bir satirindan
#    geldigi icin, o satirin gradyanina KATKI OLARAK eklenir (ayni karakter birden
#    fazla yerde kullanilmis olabilir, o yuzden += ile topluyoruz)
dC = torch.zeros_like(C)
for k in range(X.shape[0]):
    for j in range(BLOCK_SIZE):
        ix = X[k, j].item()
        dC[ix] += demb[k, j]

# --- Karsilastirma: elle hesapladiklarimiz PyTorch'un autograd'iyla AYNI MI? ---
def karsilastir(isim, manuel, otomatik):
    exact = torch.all(manuel == otomatik).item()
    yakin = torch.allclose(manuel, otomatik)
    max_fark = (manuel - otomatik).abs().max().item()
    print(f"{isim:10s} | tam_esit={str(exact):5s} | yakin={str(yakin):5s} | max_fark={max_fark:.2e}")

print("Manuel hesaplanan gradyanlar vs PyTorch autograd:")
karsilastir("dlogits", dlogits, logits.grad)
karsilastir("dh", dh, h.grad)
karsilastir("dW2", dW2, W2.grad)
karsilastir("db2", db2, b2.grad)
karsilastir("dpreact", dpreact, preact.grad)
karsilastir("dembflat", dembflat, embflat.grad)
karsilastir("dW1", dW1, W1.grad)
karsilastir("db1", db1, b1.grad)
karsilastir("demb", demb, emb.grad)
karsilastir("dC", dC, C.grad)
