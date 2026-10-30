"""
Hafta 8 - WaveNet benzeri hiyerarsik mimari.
Hafta 5'te 3 karakterin embedding'ini TEK SEFERDE birlestirip MLP'ye
veriyorduk. Bu hafta 8 karakterlik daha uzun bir context'i AMA AMA
asamali olarak ikiser ikiser birlestiriyoruz:
  8 karakter -> 4 "cift" -> 2 "dortlu" -> 1 "sekizli" -> tahmin
Bu, WaveNet'teki dilated causal convolution fikrinin basitlestirilmis hali.
"""
import torch
import torch.nn.functional as F

words = ["ada", "ali", "aslan", "kaya", "deniz", "elif", "can", "ece", "mert", "zeynep"]
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
vocab_size = len(stoi)
BLOCK_SIZE = 8  # Hafta 5'teki 3'ten daha UZUN bir context

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
print(f"X.shape: {X.shape} (8 karakterlik context)")

EMB_DIM = 8
HIDDEN_DIM = 16

g = torch.Generator().manual_seed(42)

def katman_olustur(fan_in, fan_out):
    # Hafta 6'daki initialization dersini burada da uyguluyoruz: kucuk olcek
    W = torch.randn((fan_in, fan_out), generator=g) * (fan_in ** -0.5)
    b = torch.zeros(fan_out)
    W.requires_grad_(True)
    b.requires_grad_(True)
    return W, b

C = torch.randn((vocab_size, EMB_DIM), generator=g, requires_grad=True)

# Asama 1: 8 karakter (EMB_DIM boyutlu) -> 4 cift (2*EMB_DIM boyutlu) -> HIDDEN_DIM
W1, b1 = katman_olustur(2 * EMB_DIM, HIDDEN_DIM)
# Asama 2: 4 "cift" (HIDDEN_DIM) -> 2 "dortlu" (2*HIDDEN_DIM) -> HIDDEN_DIM
W2, b2 = katman_olustur(2 * HIDDEN_DIM, HIDDEN_DIM)
# Asama 3: 2 "dortlu" (HIDDEN_DIM) -> 1 "sekizli" (2*HIDDEN_DIM) -> HIDDEN_DIM
W3, b3 = katman_olustur(2 * HIDDEN_DIM, HIDDEN_DIM)
# Son katman: HIDDEN_DIM -> vocab_size
W4, b4 = katman_olustur(HIDDEN_DIM, vocab_size)

params = [C, W1, b1, W2, b2, W3, b3, W4, b4]
print(f"Toplam parametre sayisi: {sum(p.nelement() for p in params)}")

def fuse(x, W, b):
    # x: (N, T, D) -> ardisik ciftleri birlestir -> (N, T//2, 2*D) -> linear+tanh -> (N, T//2, HIDDEN)
    N, T, D = x.shape
    x = x.view(N, T // 2, 2 * D)
    return torch.tanh(x @ W + b)

def forward(X):
    emb = C[X]                      # (N, 8, EMB_DIM)
    x = fuse(emb, W1, b1)            # (N, 4, HIDDEN_DIM)
    x = fuse(x, W2, b2)              # (N, 2, HIDDEN_DIM)
    x = fuse(x, W3, b3)              # (N, 1, HIDDEN_DIM)
    x = x.view(x.shape[0], -1)       # (N, HIDDEN_DIM)
    logits = x @ W4 + b4             # (N, vocab_size)
    return logits

learning_rate = 0.1
epochs = 300

for epoch in range(epochs):
    logits = forward(X)
    loss = F.cross_entropy(logits, Y)

    for p in params:
        p.grad = None
    loss.backward()

    for p in params:
        p.data -= learning_rate * p.grad

    if epoch % 30 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | loss = {loss.item():.4f}")

print("\nModelden uretilen ornek isimler:")
for _ in range(5):
    out = []
    context = [0] * BLOCK_SIZE
    while True:
        logits = forward(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0:
            break
        out.append([k for k, v in stoi.items() if v == ix][0])
        context = context[1:] + [ix]
    print("  " + "".join(out))
