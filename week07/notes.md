# Hafta 7 - Manuel Backpropagation

## Hedef
Tensor seviyesinde gradyani elle hesaplamak (autograd kullanmadan),
sonucu PyTorch'un gercek autograd'iyla karsilastirmak.

## Kaynak
Karpathy - Neural Networks: Zero to Hero, "Becoming a Backprop Ninja"

## Deney Notlari

## Sonuc

## Sonuclar
- Embedding -> concat -> tanh -> linear -> softmax+cross_entropy zincirinin
  TUM gradyanlari (dC, dW1, db1, dW2, db2 ve ara degerler) elle zincir kurali
  ile hesaplandi.
- PyTorch autograd ile karsilastirma: 10 gradyanin HEPSI yakin=True, max_fark
  1e-07 ile 1e-09 araliginda (sadece kayan nokta yuvarlama farki).
- En onemli iki ozel durum: (1) softmax+cross_entropy'nin birlesik turevi
  basitce "softmax(logits) - one_hot(Y)" olarak cikiyor, (2) embedding
  gradyani (dC) hesaplanirken ayni karakter birden fazla context'te
  kullanilmissa gradyanlar O SATIRA TOPLANARAK (+=) eklenmeli.
- Hafta 7 hedefi tamamlandi: manuel backprop, autograd ile birebir eslesti.
