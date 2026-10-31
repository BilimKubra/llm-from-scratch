# Hafta 10 - Multi-Head Attention ve Transformer Blogu

## Hedef
Multi-head attention, residual baglantilar, layer norm, feed-forward katman.
Karpathy'nin "Let's build GPT" videosunun ikinci yarisi, "Attention Is All
You Need" makalesi.

## Kaynak
Karpathy - "Let's build GPT", Vaswani ve ark. 2017

## Deney Notlari

## Sonuc

## Sonuclar
- Head, MultiHeadAttention, FeedForward, Block siniflari yazildi (nn.Module ile)
- Girdi/cikis shape'leri ayni (torch.Size([2, 8, 32])) - residual icin gerekli sart saglandi
- Toplam parametre: 12608 (4 head, 32 boyutlu embedding icin)
- Gradyan akisi dogrulandi: ln1.weight.grad None degil
- Onemli kavram: residual baglantilar (x = x + ...) Hafta 6'daki vanishing
  gradient sorununa bir baska cozum - cok derin aglarda bile gradyan "kisa
  yoldan" geriye akabiliyor
- Hafta 10 hedefi tamamlandi: gercek bir Transformer blogu calisir halde
