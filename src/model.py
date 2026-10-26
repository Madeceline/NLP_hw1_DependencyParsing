'随机初始化的 MLP 基线，以及 BiLSTM 与双仿射评分组成的依存分析模型。'
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class ModelConfig:
    kind: str = "biaffine"
    use_pos: bool = True
    word_dim: int = 100
    pos_dim: int = 32
    hidden: int = 200
    layers: int = 2
    arc_dim: int = 300
    rel_dim: int = 100
    dropout: float = 0.33


class Biaffine(nn.Module):
    def __init__(self, dim, outputs=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, dim + 1, dim + 1))
        nn.init.xavier_uniform_(self.weight)

    @staticmethod
    def bias(x):
        return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)

    def forward(self, dependent, head, paired=False):
        d, h = self.bias(dependent), self.bias(head)
        if paired:
            return torch.einsum("bxi,oij,bxj->bxo", d, self.weight, h)
        return torch.einsum("bxi,oij,byj->bxyo", d, self.weight, h)


def mlp(input_dim, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout))


class Parser(nn.Module):
    def __init__(self, vocabs, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.kind not in ("baseline", "biaffine"):
            raise ValueError(c.kind)
        self.word_embedding = nn.Embedding(len(vocabs["word"]), c.word_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(len(vocabs["pos"]), c.pos_dim, padding_idx=0) if c.use_pos else None
        self.dropout = nn.Dropout(c.dropout)
        input_dim = c.word_dim + (c.pos_dim if c.use_pos else 0)
        if c.kind == "biaffine":
            self.encoder = nn.LSTM(input_dim, c.hidden, c.layers, batch_first=True,
                                   bidirectional=True, dropout=c.dropout if c.layers > 1 else 0)
            encoded_dim = c.hidden * 2
        else:
            self.encoder = mlp(input_dim, c.hidden, c.dropout)
            encoded_dim = c.hidden
        self.arc_dep = mlp(encoded_dim, c.arc_dim, c.dropout)
        self.arc_head = mlp(encoded_dim, c.arc_dim, c.dropout)
        self.rel_dep = mlp(encoded_dim, c.rel_dim, c.dropout)
        self.rel_head = mlp(encoded_dim, c.rel_dim, c.dropout)
        n_rel = len(vocabs["rel"])
        if c.kind == "biaffine":
            self.arc_score = Biaffine(c.arc_dim)
            self.rel_score = Biaffine(c.rel_dim, n_rel)
        else:
            # 使用加性成对 MLP 评分，不包含循环层或双线性项。
            self.arc_score = nn.Linear(c.arc_dim, 1)
            self.rel_score = nn.Linear(c.rel_dim, n_rel)

    def forward(self, batch):
        x = self.word_embedding(batch["words"])
        if self.pos_embedding is not None:
            x = torch.cat([x, self.pos_embedding(batch["pos"])], dim=-1)
        x = self.dropout(x)
        if self.config.kind == "biaffine":
            packed = pack_padded_sequence(x, batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.encoder(packed)
            encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.size(1))
        else:
            encoded = self.encoder(x)
        encoded = self.dropout(encoded)
        d, h = self.arc_dep(encoded), self.arc_head(encoded)
        if self.config.kind == "biaffine":
            arcs = self.arc_score(d, h).squeeze(-1)
        else:
            arcs = self.arc_score(torch.tanh(d[:, :, None] + h[:, None, :])).squeeze(-1)
        arcs = arcs.masked_fill(~batch["mask"][:, None, :], -torch.inf)
        diagonal = torch.eye(arcs.size(1), dtype=torch.bool, device=arcs.device)
        arcs = arcs.masked_fill(diagonal[None], -torch.inf)
        return arcs, (self.rel_dep(encoded), self.rel_head(encoded))

    def relation_logits(self, features, heads):
        d, all_heads = features
        selected = all_heads.gather(1, heads.clamp_min(0)[..., None].expand(-1, -1, all_heads.size(-1)))
        if self.config.kind == "biaffine":
            return self.rel_score(d, selected, paired=True)
        return self.rel_score(torch.tanh(d + selected))

    def loss(self, batch, arcs, features, relation_weight=1.0):
        mask = batch["dep_mask"]
        arc_loss = F.cross_entropy(arcs[mask], batch["heads"][mask])
        logits = self.relation_logits(features, batch["heads"])
        rel_loss = F.cross_entropy(logits[mask], batch["rels"][mask])
        return arc_loss + relation_weight * rel_loss
