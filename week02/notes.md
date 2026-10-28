# Hafta 2 - Otomatik Turev (Autograd)

## Hedef
Computation graph, backpropagation, chain rule kavramlarini anlamak ve
Karpathy'nin micrograd yaklasimini sifirdan yeniden yazmak.

## Kaynak
Karpathy - Neural Networks: Zero to Hero, "micrograd" videosu

## Deney Notlari

## Sonuc

## Egitim Sonucu (train_demo.py)
- 2 girdi -> 4 -> 4 -> 1 mimarili MLP, 4 ornekli XOR-benzeri bir veri setinde egitildi
- Loss 100 epoch'ta 5.2254 -> 0.0500'e dustu
- Son tahminler hedeflere yakin: ornegin [1,1] icin tahmin 0.87 (hedef 1.0)
- Sonuc: Value sinifi + backward() zinciri gercekten calisiyor, gradient descent ile
  bir ag ogrenebiliyor. Hafta 2 hedefi (micrograd'i sifirdan yazmak) tamamlandi.
