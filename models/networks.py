import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rff import *


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=4,
        dim_ouput=3,
        num_inds=128,
        dim_hidden=128,
        num_heads=4,
        ln=True,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.proj = nn.Linear(dim_input, dim_hidden)
        self.dec = nn.Sequential(
            nn.Softplus(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Softplus(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Softplus(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Softplus(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Softplus(),
            nn.Linear(dim_hidden, dim_ouput),
        )

    def forward(self, X):
        G, _ = self.enc(X).max(dim=1, keepdim=True)
        X = self.proj(X) + G         
        X = self.dec(X)
        return X


class MaxBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(dim=1, keepdim=True)
        x = self.proj(x - xm)
        return x


class Encoder(nn.Module):
    def __init__(self, x_dim, d_dim, z1_dim):
        super().__init__()
        self.phi = nn.Sequential(
            MaxBlock(x_dim, d_dim),
            nn.Tanh(),
            MaxBlock(d_dim, d_dim),
            nn.Tanh(),
            MaxBlock(d_dim, d_dim),
            nn.Tanh(),
        )
        self.ro = nn.Sequential(
            nn.Linear(d_dim, d_dim),
            nn.Tanh(),
            nn.Linear(d_dim, z1_dim),
        )

    def forward(self, x):
        x = self.phi(x)
        x, _ = x.max(dim=1)
        z1 = self.ro(x)
        return z1


class Decoder(nn.Module):
    def __init__(self, x_dim=3, z1_dim=256, z2_dim=4, h_dim=512):
        super().__init__()
        self.fc = nn.Linear(z1_dim, h_dim)
        self.fu = nn.Linear(z2_dim, h_dim, bias=False)
        self.dec = nn.Sequential(
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, x_dim),
        )

    def forward(self, z1, z2):
        x = self.fc(z1) + self.fu(z2)
        o = self.dec(x)
        return o


class Generator(nn.Module):
    def __init__(self, x_dim=3, d_dim=256, z1_dim=256, z2_dim=4):
        super().__init__()
        self.z2_dim = z2_dim
        self.enc = Encoder(x_dim, d_dim, z1_dim)
        self.dec = Decoder(x_dim, z1_dim, z2_dim)

    def forward(self, x):
        z1 = self.enc(x[:,:,:3]).unsqueeze(dim=1)
        o = self.dec(z1, x)
        return o
