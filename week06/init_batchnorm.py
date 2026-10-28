"""
Hafta 6 - Initialization ve BatchNorm'un aktivasyonlara etkisi.
Hafta 5'teki modelde W1/W2'yi "torch.randn" ile (olcek=1.0) baslattik.
Bu hafta bunun ne kadar KOTU bir fikir oldugunu gosterip duzeltiyoruz.
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
EMB_DIM = 4
HIDDEN_DIM = 64  # bilerek buyuk tutuyoruz, sorunu daha net gormek icin

def rapor(h, baslik):
    # h: tanh sonrasi aktivasyonlar (N, HIDDEN_DIM)
    doygun = (h.abs() > 0.99).float().mean().item() * 100
    print(f"{baslik}: ortalama={h.mean():.4f}, std={h.std():.4f}, doygun_yuzde={doygun:.1f}%")

# --- Deneme 1: KOTU initialization (standart randn, olcek=1.0) ---
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, EMB_DIM), generator=g)
W1_kotu = torch.randn((BLOCK_SIZE * EMB_DIM, HIDDEN_DIM), generator=g)  # olcek=1.0 - COK BUYUK
b1_kotu = torch.randn(HIDDEN_DIM, generator=g)

emb = C[X].view(X.shape[0], -1)
h_kotu = torch.tanh(emb @ W1_kotu + b1_kotu)
rapor(h_kotu, "KOTU init (olcek=1.0)")

# --- Deneme 2: DUZELTILMIS initialization (W1'i kucuk bir sayiyla carp) ---
# Fikir: girdi boyutu buyudukce agirliklari daha kucuk baslatmaliyiz
# (Kaiming/Xavier initialization'in temel mantigi)
W1_iyi = W1_kotu * 0.2
b1_iyi = b1_kotu * 0.0  # bias'i sifirdan baslatmak da yaygin bir pratik

h_iyi = torch.tanh(emb @ W1_iyi + b1_iyi)
rapor(h_iyi, "IYI init (olcek=0.2, bias=0)")

# --- BatchNorm: her ornegin degil, HER BOYUTUN kendi ortalama/std'siyle normalize etmek ---
# Boylece baslangic olceginden bagimsiz, her zaman "saglikli" bir dagilim elde ederiz
pre_bn = emb @ W1_kotu + b1_kotu  # KOTU agirliklarla bile
bn_mean = pre_bn.mean(0, keepdim=True)
bn_std = pre_bn.std(0, keepdim=True)
pre_bn_normalized = (pre_bn - bn_mean) / bn_std
h_bn = torch.tanh(pre_bn_normalized)
rapor(h_bn, "BatchNorm (KOTU agirliklarla bile)")
