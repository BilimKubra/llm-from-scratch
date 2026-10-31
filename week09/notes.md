# Hafta 9 - Self-Attention

## Hedef
Query/key/value, dot-product attention, scaling.
Karpathy'nin "Let's build GPT" videosunun ilk yarisi.

## Kaynak
Karpathy - "Let's build GPT: from scratch, in code, spelled out."

## Deney Notlari

## Sonuc

## Sonuclar
- Versiyon 1 (sabit maskelenmis ortalama) ve Versiyon 2 (softmax ile ayni sey)
  MATEMATIKSEL OLARAK AYNI cikti (torch.allclose = True)
- Versiyon 3 (gercek q/k/v self-attention): causal mask dogru calisti (ust
  ucgen hep 0), ama agirliklar artik UNIFORM DEGIL - ornegin row 2: [0.5482,
  0.4518] yerine sabit ortalamadaki [0.5, 0.5]
- Scaling (head_size**-0.5) onemli: head_size buyudukce q@k'nin varyansi
  patlayabilir, bu da softmax'i "sivri" (bir tek degere kilitlenmis) hale
  getirip ogrenmeyi zorlastirir - olcekleme bunu onluyor
- Hafta 9 hedefi tamamlandi: uniform ortalamadan ogrenilebilir attention'a gecis
