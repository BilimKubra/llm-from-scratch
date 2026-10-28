# Hafta 3 - PyTorch Temelleri

## Hedef
Tensor mantigi, broadcasting, matrix multiplication, autograd'i PyTorch'un
kendi API'siyle anlamak (Hafta 2'de bunu elle yazmistik, simdi gercegini kullaniyoruz).

## Kaynak
PyTorch resmi tensor/autograd dokumantasyonu

## Deney Notlari

## Sonuc

## Sonuclar
- tensors.py: tek ornek icin dot product + autograd, Hafta 1'deki elle hesaplamayla
  birebir tutarli (z=0.2, aynı girdi/agirlik/bias degerleriyle)
- matmul.py: 4 ornegi AYNI ANDA (X @ W + b) isledik - GPU'larin neden hizli oldugunun
  temel sebebi bu. W.grad artik 4 ornekten gelen gradyanlarin TOPLAMI.
- Hafta 3 hedefi tamamlandi: tensor mantigi, broadcasting, matmul, autograd
