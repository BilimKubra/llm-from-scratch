# Hafta 5 - Embedding ve MLP

## Hedef
Embedding lookup, context window, hyperparameter tuning.
Karpathy'nin "makemore" serisinin 2. videosu (Bengio ve ark. 2003 makalesine dayanir).

## Kaynak
Karpathy - Neural Networks: Zero to Hero, "makemore" (MLP videosu)

## Deney Notlari

## Sonuc

## Sonuclar
- Loss 6.7230 -> 0.7437'ye dustu (300 epoch)
- Model "elif" kelimesini TAM OLARAK uretti - bu ezberleme (overfitting) belirtisi
- Sebep: 565 parametre, 50 egitim ornegi - model veri setinden "daha buyuk"
- Hyperparameter tuning notu: EMB_DIM veya HIDDEN_DIM dusurulerek, veya L2
  regularization eklenerek ezberleme azaltilabilir - gercek veri setinde
  (binlerce kelime) bu sorun daha az gorulur cunku model ezberleyemez.
- Hafta 5 hedefi tamamlandi: embedding lookup + context window + MLP calisir
  halde, ezberleme/genelleme ayrimi gozlemlendi.
