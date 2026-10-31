# Hafta 11 - GPT'yi Egitmek

## Hedef
Tam GPT mimarisi, egitim dongusu, kucuk bir veri seti uzerinde egitim.
nanoGPT reposuna dayanir.

## Kaynak
Karpathy - nanoGPT, "Let's build GPT" videosunun sonu

## Deney Notlari

## Sonuc

## Sonuclar
- Vocab=26, toplam veri=346 karakter, model parametre sayisi=40090
- train_loss: 3.4280 -> 0.2191 (model ogreniyor)
- val_loss: 3.1155 -> 6.4232 (AYNI ANDA yukseliyor - klasik overfitting)
- Uretilen metinde egitim corpus'undan BIREBIR kopyalanmis bolumler var
  ("ada ali aslan kaya deniz elif can ece mert zeynep bu")
- Yorum: 40k parametreye 346 karakterlik veri COK AZ - model ezberliyor,
  genellemiyor. Bu, Hafta 5/8'deki aynı sorunun en buyuk olcekte gorulmesi
  ve gercek GPT'lerin neden milyarlarca token'la egitildiginin somut kaniti.
- Mimari dogru calisiyor (train_loss dususu bunu kanitliyor), sorun veri
  miktarinda - capstone'da buna cozum (daha fazla veri, regularization,
  daha kucuk model) denenebilir.
- Hafta 11 hedefi tamamlandi: tam GPT mimarisi sifirdan yazildi ve calisti.
