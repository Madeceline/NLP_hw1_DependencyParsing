import copy

import pytest
import torch

from src.data import ParserDataset, Sentence, build_vocabs, collate
from src.engine import attachment_scores, infer, load_parser, train
from src.model import Biaffine, ModelConfig, Parser


def sentences():
    return [Sentence(["1\tI\t_\tPRON\t_\t_\t2\tnsubj\t_\t_".split("\t"),
                      "2\trun\t_\tVERB\t_\t_\t0\troot\t_\t_".split("\t")]),
            Sentence(["1\tRun\t_\tVERB\t_\t_\t0\troot\t_\t_".split("\t")])]


def small_config(kind="biaffine", use_pos=True):
    return ModelConfig(kind=kind, use_pos=use_pos, word_dim=12, pos_dim=5, hidden=12,
                       layers=1, arc_dim=16, rel_dim=8, dropout=0.0)


@pytest.mark.parametrize("kind,use_pos", [("baseline", True), ("biaffine", False), ("biaffine", True)])
def test_forward_backward_padding_and_no_gold_leak(kind, use_pos):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    s = sentences()
    v = build_vocabs(s)
    model = Parser(v, small_config(kind, use_pos))
    dataset = ParserDataset(s, v)
    batch = collate([dataset[0], dataset[1]])
    arcs, features = model(batch)
    assert torch.isneginf(arcs[1, :, 2]).all()  # 填充位置不能作为候选父节点
    assert torch.isneginf(arcs.diagonal(dim1=1, dim2=2)).all()
    loss = model.loss(batch, arcs, features)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.word_embedding.weight.grad.abs().sum() > 0
    model.eval()
    alone, _ = model(collate([dataset[1]]))
    assert torch.allclose(arcs[1, :2, :2], alone[0], atol=1e-6)
    corrupt = copy.deepcopy(s)
    for sentence in corrupt:
        for row in sentence.tokens:
            row[6] = row[7] = "_"
    original_predictions, _ = infer(model, s, v)
    blank_predictions, _ = infer(model, corrupt, v)
    assert original_predictions == blank_predictions
    for heads, rels in blank_predictions:
        assert all((h == 0) == (r == "root") for h, r in zip(heads[1:], rels[1:]))


def test_biaffine_algebra():
    scorer = Biaffine(2, 3)
    d, h = torch.randn(2, 4, 2), torch.randn(2, 4, 2)
    pairwise = scorer(d, h)
    paired = scorer(d, h, paired=True)
    for b in range(2):
        for i in range(4):
            manual = torch.stack([scorer.bias(d)[b, i] @ w @ scorer.bias(h)[b, i] for w in scorer.weight])
            assert torch.allclose(paired[b, i], manual, atol=1e-6)
            assert torch.allclose(pairwise[b, i, i], paired[b, i], atol=1e-6)


def test_metric_punctuation_and_subtypes():
    s = Sentence(["1\tA\t_\tNOUN\t_\t_\t0\troot\t_\t_".split("\t"),
                  "2\tB\t_\tNOUN\t_\t_\t1\tnmod:poss\t_\t_".split("\t"),
                  "3\t.\t_\tPUNCT\t_\t_\t1\tpunct\t_\t_".split("\t")])
    scores = attachment_scores([s], [([-1, 0, 1, 2], ["<ROOT>", "root", "nmod", "punct"])])
    assert scores["all_full"]["LAS"] == pytest.approx(100 / 3)
    assert scores["all_coarse"]["LAS"] == pytest.approx(200 / 3)
    assert scores["no_punct_full"]["UAS"] == 100


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    s = sentences()
    v = build_vocabs(s)
    config = dict(seed=42, device="cpu", threads=2, batch_size=2, epochs=2, patience=5,
                  learning_rate=0.01, gradient_clip=5.0, relation_weight=1.0, word_dropout=0.1)
    model_config = small_config()
    train("tiny", s, s, v, config, model_config, tmp_path / "continuous")
    train("tiny", s, s, v, {**config, "epochs": 1}, model_config, tmp_path / "resumed")
    train("tiny", s, s, v, config, model_config, tmp_path / "resumed", resume=True)
    _, _, a = load_parser(tmp_path / "continuous/last.pt")
    _, _, b = load_parser(tmp_path / "resumed/last.pt")
    assert a["epoch"] == b["epoch"] == 2
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name])


def test_tiny_dataset_can_be_overfit():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    s = sentences()
    v = build_vocabs(s)
    model = Parser(v, small_config())
    ds = ParserDataset(s, v)
    batch = collate([ds[0], ds[1]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(60):
        model.train()
        optimizer.zero_grad()
        arcs, features = model(batch)
        loss = model.loss(batch, arcs, features)
        loss.backward()
        optimizer.step()
    predictions, _ = infer(model, s, v)
    assert attachment_scores(s, predictions)["all_full"]["LAS"] == 100
