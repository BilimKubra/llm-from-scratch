# Hafta 4 - Dil Modelleme (Bigram)

## Hedef
Bigram model, negatif log likelihood, karakter seviyesinde modelleme.
Karpathy'nin "makemore" serisinin ilk videosuna dayaniyor.

## Kaynak
Karpathy - Neural Networks: Zero to Hero, "makemore" (ilk video)

## Deney Notlari

## Sonuc

## Sonuclar
- Sayma tabanli model (bigram_counting.py): N tablosu + Laplace smoothing ile
  olasiliklar hesaplandi, NLL olculdu, ornekleme ile isimler uretildi.
- Noral model (bigram_neural.py): tek katmanli agirlik matrisi + softmax +
  gradient descent ile AYNI bigram istatistiklerini ogrendi.
  Loss 3.2243 -> 1.2856'ya dustu (100 epoch).
- Onemli kavram: "sayma tabanli" ve "sinir agi tabanli" modeller matematiksel
  olarak esdeger - biri frekanslari direkt sayiyor, digeri gradient descent ile
  aynı frekanslara YAKINSIYOR. Bu, klasik istatistiksel modellerden sinir
  aglarina gecisin en temel ornegi.
- Hafta 4 ve Faz 1 (Hafta 1-4) TAMAMLANDI.
