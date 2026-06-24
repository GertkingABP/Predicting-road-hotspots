"""
model.py — архитектура нейросетевой модели GRU+GCN
"""
import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W = nn.Linear(in_d, out_d)

    def forward(self, X, A):
        return torch.relu(A @ self.W(X))


class GRU_GCN(nn.Module):
    """
    Пространственно-временная нейросетевая модель.
    Вход:
        X : (B, H, N, F) — признаки трафика
        A : (N, N)       — нормализованная матрица смежности
    Выход:
        p : (B, N)       — вероятность аварии в каждом узле
    """
    def __init__(self, F_in=10, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.gru  = nn.GRU(F_in, hidden, batch_first=True)
        self.gcn1 = GCN(hidden, hidden)
        self.gcn2 = GCN(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, X, A):
        B, H, N, F = X.shape
        Xr = X.permute(0, 2, 1, 3).reshape(B * N, H, F)
        _, h = self.gru(Xr)
        ht = h[0].view(B, N, self.hidden)
        z  = self.gcn1(ht, A)
        z  = self.gcn2(z,  A)
        return self.head(z).squeeze(-1)   # (B, N)


def load_model(pt_path: str, device: str = "cpu") -> GRU_GCN:
    model = GRU_GCN()
    state = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
