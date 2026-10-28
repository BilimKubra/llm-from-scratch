# Hafta 6 - Egitim Mekanigi

## Hedef
Initialization, BatchNorm, gradyan istatistiklerini okumak.
Karpathy'nin "makemore" serisinin 3. videosuna dayanir (Activations & Gradients).

## Kaynak
Karpathy - Neural Networks: Zero to Hero, "makemore" (BatchNorm videosu)

## Deney Notlari

## Sonuc

## Sonuclar
- Kotu init (olcek=1.0): %49.4 noron doymus - yarisi pratikte olu, ogrenemiyor
- Iyi init (olcek=0.2, bias=0): %1.4 doymus - buyuk iyilesme, tek degisken agirlik olcegi
- BatchNorm (kotu agirliklarla bile): %0.7 doymus - BatchNorm, init olceginden
  BAGIMSIZ olarak saglikli dagilim sagliyor
- Sonuc: initialization onemli ama BatchNorm bunu "unutmaya" izin veren bir
  guvenlik agi gibi davraniyor. Hafta 1'deki sigmoid doygunluk gozlemiyle
  dogrudan baglantili.
- Hafta 6 hedefi tamamlandi.

## Sonuclar
- Kotu init (olcek=1.0): %49.4 noron doymus - yarisi pratikte olu, ogrenemiyor
- Iyi init (olcek=0.2, bias=0): %1.4 doymus - buyuk iyilesme, tek degisken agirlik olcegi
- BatchNorm (kotu agirliklarla bile): %0.7 doymus - BatchNorm, init olceginden
  BAGIMSIZ olarak saglikli dagilim sagliyor
- Sonuc: initialization onemli ama BatchNorm bunu "unutmaya" izin veren bir
  guvenlik agi gibi davraniyor. Hafta 1'deki sigmoid doygunluk gozlemiyle
  dogrudan baglantili.
- Hafta 6 hedefi tamamlandi.
