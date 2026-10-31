"""
Hafta 10 - Multi-head attention + Transformer blogu.
Hafta 9'da TEK BIR attention head yazmistik. Simdi birden fazlasini PARALEL
calistirip (multi-head), residual baglantilar + layer norm + feed-forward
ekleyerek gercek bir Transformer "block"u kuruyoruz.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Hafta 9'daki self-attention mantiginin tek bir 'head' olarak sinifa dokulmus hali."""
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Birden fazla Head'i PARALEL calistirip sonuclari birlestirir (concat).
    Fikir: her head farkli bir 'iliski turunu' ogrenebilir (biri yakin baglam,
    biri uzak baglam, biri sozdizimi gibi)."""
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Attention'dan sonra her token'in kendi icinde ek islem yapmasini saglayan
    basit bir MLP. Attention 'token'lar arasi iletisim', bu ise 'token'in kendi
    icinde dusunmesi' gibi dusunulebilir."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Gercek bir Transformer blogu: attention + feedforward, RESIDUAL baglantilar
    ve LAYER NORM ile. Residual (x = x + ...) sayesinde cok derin aglarda bile
    gradyan akisi kaybolmuyor - Hafta 6'daki 'doygunluk/vanishing gradient'
    sorununa bir baska cozum."""
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, num_heads, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-norm + residual: once normalize et, isle, SONUCU x'e EKLE (cikarma degil)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, C = 2, 8, 32  # 2 ornek, 8 token, 32 boyutlu embedding
    x = torch.randn(B, T, C)

    block = Block(n_embd=C, num_heads=4, block_size=T)
    out = block(x)

    print(f"Girdi shape:  {x.shape}")
    print(f"Cikis shape:  {out.shape}")
    print(f"Shape'ler ayni mi (residual icin gerekli)? {x.shape == out.shape}")
    print(f"\nToplam parametre sayisi: {sum(p.numel() for p in block.parameters())}")

    # Gradyan akisinin calistigini dogrula
    loss = out.sum()
    loss.backward()
    print(f"\nGradyan akisi calisti mi? ln1.weight.grad None degil: {block.ln1.weight.grad is not None}")
