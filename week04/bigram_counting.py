"""
Hafta 4 - Sayma tabanli (counting-based) bigram dil modeli.
Fikir: "X harfinden sonra Y harfi kac kere geldi" diye sayip, bu sayilardan
olasilik cikariyoruz. Bu, egitimsiz en basit dil modeli turu.
"""
import torch

# Kucuk bir isim veri seti (gercek makemore'da binlerce isim var, biz kucuk basliyoruz)
words = ["ada", "ali", "aslan", "kaya", "deniz", "elif", "can", "ece", "mert", "zeynep"]

# Karakterleri sayilara cevirmek icin vocab olustur
# "." hem kelime baslangicini hem bitisini temsil eden ozel bir token
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

print("Vocab (karakter -> sayi):", stoi)

# Bigram sayim tablosu: N[i][j] = i. karakterden sonra j. karakterin kac kere geldigi
N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Sayimlari olasiliklara cevir (her satiri kendi toplamina bolup normalize et)
# +1 ekliyoruz (Laplace smoothing) - hic gormedigimiz bir bigram'a 0 olasilik vermemek icin
P = (N + 1).float()
P = P / P.sum(dim=1, keepdim=True)

# --- Negatif log likelihood: modelin veriye ne kadar "iyi uydugunun" olcusu ---
# Dusuk NLL = model bu veriyi gormesini daha olasi buluyor = daha iyi model
log_likelihood = 0.0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

nll = -log_likelihood / n
print(f"\nNegatif log likelihood (ortalama, dusuk = iyi): {nll.item():.4f}")

# --- Modelden yeni "isimler" uret (olasiliklara gore ornekleme yaparak) ---
print("\nModelden uretilen ornek isimler:")
g = torch.Generator().manual_seed(42)
for _ in range(5):
    out = []
    ix = 0  # "." ile basla
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("  " + "".join(out))
