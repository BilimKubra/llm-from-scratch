"""
Hafta 11 - Tam GPT mimarisi, sifirdan.
Hafta 1-10'da ogrendigimiz HER SEYI (forward pass, autograd, embedding,
initialization, backprop, attention, transformer block) burada birlestiriyoruz.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# --- Veri: kucuk bir karakter seviyesinde metin korpusu ---
text = (
    "yz50 programina basvurdum ve kabul sonucunu beklemeden kendi basima "
    "oniki haftalik bir ogrenme surecine basladim. amacim modelleri sadece "
    "api uzerinden kullanan biri olmaktan cikip neden oyle davrandiklarini "
    "satir satir takip edebilen biri olmak. ada ali aslan kaya deniz elif "
    "can ece mert zeynep bu surecte sifirdan bir dil modeli insa ediyorum."
)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Vocab boyutu: {vocab_size}, toplam karakter: {len(data)}")

# --- Hiperparametreler ---
BLOCK_SIZE = 16   # context uzunlugu
BATCH_SIZE = 16
N_EMBD = 32
N_HEAD = 4
N_LAYER = 3
LR = 1e-2
MAX_ITERS = 500

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.ReLU(), nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(N_HEAD, N_EMBD // N_HEAD)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)                                  # (B,T,N_EMBD)
        pos_emb = self.pos_emb(torch.arange(T))                        # (T,N_EMBD)
        x = tok_emb + pos_emb                                           # her token "ne oldugunu" VE "nerede oldugunu" biliyor
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                           # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


model = GPT()
print(f"Toplam parametre sayisi: {sum(p.numel() for p in model.parameters())}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for it in range(MAX_ITERS):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 50 == 0 or it == MAX_ITERS - 1:
        xv, yv = get_batch("val")
        with torch.no_grad():
            _, val_loss = model(xv, yv)
        print(f"Iter {it:4d} | train_loss={loss.item():.4f} | val_loss={val_loss.item():.4f}")

print("\nModelden uretilen metin:")
context = torch.zeros((1, 1), dtype=torch.long)  # "." yerine ilk karakterden basla
generated = model.generate(context, max_new_tokens=200)
print(decode(generated[0].tolist()))
