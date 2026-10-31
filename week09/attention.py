"""
Hafta 9 - Self-attention'a adim adim giris.
Karpathy'nin kendi pedagojik sirasini takip ediyoruz:
  1) Sadece gecmise bakarak ORTALAMA almak (en basit "iletisim" sekli)
  2) Ayni seyi matris carpimiyla (masked ortalama) yapmak
  3) GERCEK self-attention: her token, gecmisi SABIT agirliklarla degil,
     ONEMINE gore (query/key uyumuna gore) agirliklandiriyor
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)
B, T, C = 1, 8, 16  # Batch=1, Time (dizi uzunlugu)=8, Channel (embedding boyutu)=16
x = torch.randn(B, T, C)

print(f"x.shape: {x.shape}  (1 ornek, 8 token, 16 boyutlu embedding)")

# --- Versiyon 1: Her token, SADECE KENDI VE GECMISININ ortalamasini alsin ---
# Bu "causal" (nedensel) bir iletisim: gelecegi gormemeli, sadece gecmisi.
wei_v1 = torch.tril(torch.ones(T, T))              # alt ucgen matris: j<=i ise 1
wei_v1 = wei_v1 / wei_v1.sum(1, keepdim=True)       # her satiri normalize et (ortalama icin)
out_v1 = wei_v1 @ x                                  # (T,T) @ (B,T,C) -> (B,T,C)
print("\nVersiyon 1 (sabit ortalama) agirlik matrisi (ilk 4 satir/sutun):")
print(wei_v1[:4, :4])

# --- Versiyon 2: Ayni sey ama softmax ile (ileride ogrenilebilir hale gelecek) ---
tril = torch.tril(torch.ones(T, T))
wei_v2 = torch.zeros((T, T))
wei_v2 = wei_v2.masked_fill(tril == 0, float("-inf"))  # gelecek pozisyonlari -inf yap
wei_v2 = F.softmax(wei_v2, dim=-1)                      # softmax(-inf)=0, softmax(0,0,0,...)=duzgun ortalama
out_v2 = wei_v2 @ x
print("\nVersiyon 1 ve Versiyon 2 ayni mi?", torch.allclose(out_v1, out_v2))

# --- Versiyon 3: GERCEK self-attention (query, key, value ile) ---
head_size = 16
key = torch.nn.Linear(C, head_size, bias=False)
query = torch.nn.Linear(C, head_size, bias=False)
value = torch.nn.Linear(C, head_size, bias=False)

k = key(x)     # (B, T, head_size)   - "ben neyim" bilgisi
q = query(x)   # (B, T, head_size)   - "ben ne ariyorum" bilgisi
v = value(x)   # (B, T, head_size)   - "eger secilirsem, verecegim bilgi"

# Her token'in sorgusu (q), her token'in anahtariyla (k) ne kadar "uyustugunu" olc
wei_v3 = q @ k.transpose(-2, -1)        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
wei_v3 = wei_v3 * (head_size ** -0.5)   # OLCEKLEME: head_size buyudukce varyans patlamasin diye
wei_v3 = wei_v3.masked_fill(tril == 0, float("-inf"))  # causal mask: gelecegi gorme
wei_v3 = F.softmax(wei_v3, dim=-1)      # artik agirliklar SABIT degil, OGRENILEBILIR

out_v3 = wei_v3 @ v

print("\nVersiyon 3 (gercek self-attention) agirlik matrisi (ilk ornek, ilk 4 satir/sutun):")
print(wei_v3[0, :4, :4])
print("\nVersiyon 3 cikisi shape:", out_v3.shape)
print("\nOnemli fark: Versiyon 1/2'de her satirin agirliklari SABIT (uniform ortalama).")
print("Versiyon 3'te her satirin agirliklari q@k'ya gore DEGISKEN - model hangi")
print("gecmis token'in daha 'onemli' oldugunu OGRENEBILIYOR.")
