"""
Hafta 5 - Embedding + MLP tabanli karakter seviyesinde dil modeli.
Hafta 4'teki bigram sadece 1 onceki karaktere bakiyordu. Burada BLOCK_SIZE
kadar onceki karaktere bakiyoruz ve her karakteri once kucuk bir vektore
(embedding) ceviriyoruz - bu, Bengio ve ark. 2003 makalesinin temel fikri.
"""
import torch
import torch.nn.functional as F

words = ["ada", "ali", "aslan", "kaya", "deniz", "elif", "can", "ece", "mert", "zeynep"]

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

BLOCK_SIZE = 3  # kac onceki karaktere bakacagiz

# --- Veri setini olustur: (context, hedef) ciftleri ---
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE  # basta hepsi "." (padding)
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # pencereyi bir kaydir
    return torch.tensor(X), torch.tensor(Y)

X, Y = build_dataset(words)
print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")
print(f"Ornek: context={X[0].tolist()} -> hedef={Y[0].item()} ('{itos[Y[0].item()]}')")

# --- Model parametreleri ---
EMB_DIM = 4     # her karakter kac boyutlu bir vektorle temsil edilecek
HIDDEN_DIM = 16

g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, EMB_DIM), generator=g, requires_grad=True)       # embedding tablosu
W1 = torch.randn((BLOCK_SIZE * EMB_DIM, HIDDEN_DIM), generator=g, requires_grad=True)
b1 = torch.randn(HIDDEN_DIM, generator=g, requires_grad=True)
W2 = torch.randn((HIDDEN_DIM, vocab_size), generator=g, requires_grad=True)
b2 = torch.randn(vocab_size, generator=g, requires_grad=True)
params = [C, W1, b1, W2, b2]
print(f"Toplam parametre sayisi: {sum(p.nelement() for p in params)}")

learning_rate = 0.1
epochs = 300

for epoch in range(epochs):
    # --- Ileri gecis ---
    emb = C[X]                              # (N, BLOCK_SIZE, EMB_DIM) - her context icin embedding'leri cek
    emb_flat = emb.view(emb.shape[0], -1)   # (N, BLOCK_SIZE*EMB_DIM) - duzlestir (concat)
    h = torch.tanh(emb_flat @ W1 + b1)      # (N, HIDDEN_DIM) - gizli katman
    logits = h @ W2 + b2                    # (N, vocab_size)
    loss = F.cross_entropy(logits, Y)       # softmax + NLL bir arada

    # --- Geri gecis ---
    for p in params:
        p.grad = None
    loss.backward()

    # --- Guncelleme ---
    for p in params:
        p.data -= learning_rate * p.grad

    if epoch % 30 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | loss = {loss.item():.4f}")

print("\nModelden uretilen ornek isimler:")
for _ in range(5):
    out = []
    context = [0] * BLOCK_SIZE
    while True:
        emb = C[torch.tensor([context])]
        emb_flat = emb.view(1, -1)
        h = torch.tanh(emb_flat @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    print("  " + "".join(out))
