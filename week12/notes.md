# Hafta 12 - Capstone: Overfitting'i Cozmeye Calismak

## Hedef
Hafta 11'de bulunan asiri ezberleme (overfitting) sorununu, daha fazla veri
ve dropout (regularization) ekleyerek azaltmaya calismak.

## Deney Notlari

## Sonuc

## Sonuclar: Hafta 11 vs Hafta 12 Karsilastirmasi
| Metrik | Hafta 11 (346 char, dropout yok) | Hafta 12 (1857 char, dropout=0.2) |
|---|---|---|
| train_loss son | 0.22 | 1.82 |
| val_loss son | 6.42 | 2.75 (en dusuk 2.10) |
| train/val farki | ~6.2 | ~0.9 |

- 5x daha fazla veri + dropout=0.2, val_loss'un KONTROLSUZCE PATLAMASINI ONLEDI
- train_loss artik sifira yakin degil (1.82) - model ARTIK EZBERLEYEMIYOR,
  bu YUZDEN daha genel bir sey ogrenmeye zorlaniyor
- Uretilen metin daha az akici ama bu beklenen: 1857 karakter gercek bir
  dil ogrenmek icin hala COK AZ (gercek GPT'ler milyarlarca token kullanir)
- ANA SONUC: overfitting/genelleme dengesi veri miktari VE regularization
  (dropout) ile dogrudan kontrol edilebilir - bu, Hafta 1'den (forward pass)
  Hafta 12'ye (tam GPT + regularization) kadar olan tum yolculugun pratik
  ozeti.
- CAPSTONE VE PROGRAM TAMAMLANDI.
