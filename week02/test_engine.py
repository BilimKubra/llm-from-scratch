"""
week02/engine.py'daki Value sinifinin urettigi gradyanlari
PyTorch'un gercek autograd'i ile karsilastiran test.
Ikisi ayni sonucu vermiyorsa, kendi backward() implementasyonumuzda
bir hata var demektir.
"""
import torch
from week02.engine import Value


def test_karsilastirma():
    # --- Bizim Value sinifimizla ---
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    e = c - d
    f = e**2
    g = f.tanh()
    g.backward()

    # --- Ayni hesap PyTorch ile ---
    at = torch.Tensor([-4.0]).double()
    at.requires_grad = True
    bt = torch.Tensor([2.0]).double()
    bt.requires_grad = True
    ct = at + bt
    dt = at * bt + bt**3
    et = ct - dt
    ft = et**2
    gt = torch.tanh(ft)
    gt.backward()

    # --- Karsilastirma ---
    tol = 1e-6
    assert abs(g.data - gt.data.item()) < tol, f"ileri gecis farkli: {g.data} vs {gt.data.item()}"
    assert abs(a.grad - at.grad.item()) < tol, f"a.grad farkli: {a.grad} vs {at.grad.item()}"
    assert abs(b.grad - bt.grad.item()) < tol, f"b.grad farkli: {b.grad} vs {bt.grad.item()}"

    print("TUM TESTLER GECTI")
    print(f"g.data={g.data:.6f} vs pytorch={gt.data.item():.6f}")
    print(f"a.grad={a.grad:.6f} vs pytorch={at.grad.item():.6f}")
    print(f"b.grad={b.grad:.6f} vs pytorch={bt.grad.item():.6f}")


if __name__ == "__main__":
    test_karsilastirma()
