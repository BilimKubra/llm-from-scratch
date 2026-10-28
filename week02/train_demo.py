"""
Hafta 2 kapanisi: MLP'yi kucuk bir ikili siniflandirma veri setinde egitmek.
Amac: Value + Neuron + Layer + MLP zincirinin GERCEKTEN ogrenebildigini gormek.
"""
from week02.nn import MLP

# Cok basit bir veri seti: 4 nokta, 2 ozellik (x1, x2), hedef: ikisinin ayni isaretli olup olmadigi
xs = [
    [1.0, 1.0],
    [1.0, -1.0],
    [-1.0, 1.0],
    [-1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # ayni isaretliyse 1, degilse -1

model = MLP(2, [4, 4, 1])  # 2 girdi -> 4 -> 4 -> 1 cikti
print(f"Toplam parametre sayisi: {len(model.parameters())}")

learning_rate = 0.05
epochs = 100

for epoch in range(epochs):
    # --- Ileri gecis: tahminleri hesapla, loss'u bul ---
    ypred = [model(x) for x in xs]
    loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, ys))

    # --- Geri gecis: gradyanlari sifirla, backward cagir ---
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # --- Guncelleme: her parametreyi gradyaninin tersi yonunde kucuk bir adim attir ---
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | loss = {loss.data:.4f}")

print("\nSon tahminler:")
for x, yt in zip(xs, ys):
    pred = model(x)
    print(f"  girdi={x}, hedef={yt}, tahmin={pred.data:.4f}")
