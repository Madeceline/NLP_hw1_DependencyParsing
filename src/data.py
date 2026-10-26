'CoNLL-U 读写、仅由训练集构建的词表，以及批次填充处理。'
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD, UNK, ROOT = "<PAD>", "<UNK>", "<ROOT>"


@dataclass
class Sentence:
    rows: list

    @property
    def tokens(self):
        return [r for r in self.rows if isinstance(r, list) and r[0].isdigit()]

    @property
    def text(self):
        for row in self.rows:
            if isinstance(row, str) and row.startswith("# text = "):
                return row[9:]
        return " ".join(r[1] for r in self.tokens)

    def __len__(self):
        return len(self.tokens)

    def gold(self):
        '将文件中的词编号映射为连续位置索引；推理过程不会调用此方法。'
        mapping = {int(r[0]): i for i, r in enumerate(self.tokens, 1)}
        mapping[0] = 0
        try:
            heads = [-1] + [mapping[int(r[6])] for r in self.tokens]
        except (ValueError, KeyError) as exc:
            raise ValueError("基本依存父节点标注缺失或无效") from exc
        return heads, [ROOT] + [r[7] for r in self.tokens]


def read_conllu(path):
    sentences, rows = [], []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines() + [""]:
        if not line.strip():
            if rows:
                sentence = Sentence(rows)
                ids = [int(r[0]) for r in sentence.tokens]
                if not ids or len(ids) != len(set(ids)) or ids != sorted(ids):
                    raise ValueError("句子必须包含唯一且递增的整数词编号")
                sentences.append(sentence)
                rows = []
        elif line.startswith("#"):
            rows.append(line)
        else:
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(f"每个条目应有 10 列： {line!r}")
            rows.append(fields)
    return sentences


def write_predictions(sentences, predictions, path):
    '保留表层元信息和多词条目，清除与预测结果不一致的增强依存信息。\n\n空节点仅用于增强依存图，因此输出基本依存树时将其删除。\n若空节点的 DEPS 字段为下划线，则不符合 CoNLL-U 格式规范。'
    from .decoder import validate_tree
    if len(sentences) != len(predictions):
        raise ValueError("句子数与预测结果数量不一致")
    lines = []
    for sentence, (heads, relations) in zip(sentences, predictions):
        validate_tree(heads)
        if len(heads) != len(sentence) + 1 or len(relations) != len(heads):
            raise ValueError("预测结果长度不一致")
        ids = [0] + [int(r[0]) for r in sentence.tokens]
        position = {file_id: i for i, file_id in enumerate(ids)}
        for row in sentence.rows:
            if isinstance(row, str):
                lines.append(row)
                continue
            if "." in row[0]:
                continue
            row = row.copy()
            row[8] = "_"
            if row[0].isdigit():
                i = position[int(row[0])]
                row[6], row[7] = str(ids[heads[i]]), relations[i]
                if (heads[i] == 0) != (relations[i] == "root"):
                    raise ValueError("父节点为 0 当且仅当依存关系为 root")
            lines.append("\t".join(row))
        lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_vocabs(train):
    words = Counter(r[1].lower() for s in train for r in s.tokens)
    pos = sorted({r[3] for s in train for r in s.tokens})
    rel = sorted({r[7] for s in train for r in s.tokens})
    return {
        "word": {w: i for i, w in enumerate([PAD, UNK, ROOT] + sorted(words))},
        "pos": {w: i for i, w in enumerate([PAD, UNK, ROOT] + pos)},
        "rel": {w: i for i, w in enumerate(rel)},
    }


class ParserDataset(Dataset):
    def __init__(self, sentences, vocabs, labeled=True):
        self.sentences = sentences
        self.examples = []
        for index, s in enumerate(sentences):
            words = [vocabs["word"][ROOT]] + [
                vocabs["word"].get(r[1].lower(), vocabs["word"][UNK]) for r in s.tokens
            ]
            pos = [vocabs["pos"][ROOT]] + [
                vocabs["pos"].get(r[3], vocabs["pos"][UNK]) for r in s.tokens
            ]
            example = {"words": words, "pos": pos, "index": index}
            if labeled:
                heads, relations = s.gold()
                example["heads"] = heads
                # 验证集或测试集中未见于训练集的关系没有对应类别，记为 -1。
                # 该标记仅用于核查，不参与训练目标的计算。
                example["rels"] = [-1] + [vocabs["rel"].get(r, -1) for r in relations[1:]]
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(examples):
    lengths = torch.tensor([len(e["words"]) for e in examples])
    width = int(lengths.max())
    batch = {"lengths": lengths, "indices": [e["index"] for e in examples]}
    for key in ("words", "pos", "heads", "rels"):
        if key not in examples[0]:
            continue
        tensor = torch.full((len(examples), width), -1 if key in ("heads", "rels") else 0, dtype=torch.long)
        for i, example in enumerate(examples):
            tensor[i, :len(example[key])] = torch.tensor(example[key])
        batch[key] = tensor
    batch["mask"] = torch.arange(width)[None, :] < lengths[:, None]
    batch["dep_mask"] = batch["mask"].clone()
    batch["dep_mask"][:, 0] = False
    return batch


def dataset_statistics(splits, vocabs):
    summary = []
    for name, sentences in splits.items():
        tokens = [r for s in sentences for r in s.tokens]
        oov = sum(r[1].lower() not in vocabs["word"] for r in tokens)
        summary.append({"split": name, "sentences": len(sentences), "words": len(tokens),
                        "mean_length": len(tokens) / len(sentences), "max_length": max(map(len, sentences)),
                        "oov_percent": 100 * oov / len(tokens)})
    return summary
