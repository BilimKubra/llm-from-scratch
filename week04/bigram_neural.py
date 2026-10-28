"""
Hafta 4 - Noral (neural) bigram dil modeli.
Ayni bigram fikrini bu kez sayma yerine bir sinir agiyla (tek katman +
softmax) ogreniyoruz. Amac: sayma tabanli model ile NOORAL modelin
AYNI sonuca yakinsadigini gormek - bu, "istatistiksel model" ile
"sinir agi" arasindaki koprunun en basit hali.
"""
import torch
import torch.nn.functional as F

words = ["ada", "ali", "aslan", "kaya", "deniz", "elif", "can", "ece", "mert", "zeynep"]

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num_examples = xs.shape[0]
print(f"Toplam egitim ornegi: {num_examples}")

g = torch.Generator().manual_seed(42)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

learning_rate = 10.0
epochs = 100

for epoch in range(epochs):
    xenc = F.one_hot(xs, num_classes=vocab_size).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(num_examples), ys].log().mean()
    loss = loss + 0.01 * (W ** 2).mean()

    W.grad = None
    loss.backward()

    W.data -= learning_rate * W.grad

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | loss (NLL) = {loss.item():.4f}")

print("\nNoral modelden uretilen ornek isimler:")
for _ in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc @ W
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("  " + "".join(out))
