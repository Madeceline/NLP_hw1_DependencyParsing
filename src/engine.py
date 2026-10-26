'实现模型训练、断点保存、推理和依存准确率计算。'
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .data import ParserDataset, collate, read_conllu, write_predictions
from .decoder import decode_mst, validate_tree
from .model import ModelConfig, Parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 避免 cuBLAS 选择无法复现的工作区算法。
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) and k != "lengths" else v
            for k, v in batch.items()}


class LengthBatches(Sampler):
    '先打乱较大的样本池，再在池内按长度排序，降低填充带来的二次方计算开销。'
    def __init__(self, dataset, batch_size, seed):
        self.lengths = [len(e["words"]) for e in dataset.examples]
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        batches = []
        pool_size = self.batch_size * 50
        for start in range(0, len(indices), pool_size):
            pool = sorted(indices[start:start + pool_size], key=lambda i: self.lengths[i])
            batches.extend(pool[i:i + self.batch_size] for i in range(0, len(pool), self.batch_size))
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


@torch.inference_mode()
def infer(model, sentences, vocabs, device="cpu", batch_size=32):
    '仅编码词形和通用词性；不读取输入中的真实父节点和依存关系。'
    model.eval()
    dataset = ParserDataset(sentences, vocabs, labeled=False)
    # 按长度排序以减少评估时的填充，预测后恢复原始句子顺序。
    indices = sorted(range(len(dataset)), key=lambda i: len(dataset.examples[i]["words"]))
    predictions = [None] * len(dataset)
    id_to_rel = {i: label for label, i in vocabs["rel"].items()}
    root_id = vocabs["rel"]["root"]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for offset in range(0, len(indices), batch_size):
        batch = move_batch(collate([dataset[i] for i in indices[offset:offset + batch_size]]), device)
        arcs, features = model(batch)
        matrices = arcs.detach().cpu().numpy()
        all_heads = torch.zeros_like(batch["words"])
        decoded = []
        for b, length in enumerate(batch["lengths"].tolist()):
            heads = decode_mst(matrices[b, :length, :length])
            validate_tree(heads)
            decoded.append(heads)
            all_heads[b, :length] = torch.tensor(heads, device=device)
        logits = model.relation_logits(features, all_heads)
        # 显式保证父节点为 0 当且仅当依存关系为 root。
        logits[..., root_id] = logits[..., root_id].masked_fill(all_heads != 0, -torch.inf)
        labels = logits.argmax(-1).cpu().numpy()
        for b, (index, heads) in enumerate(zip(batch["indices"], decoded)):
            relations = ["<ROOT>"] + ["root" if heads[i] == 0 else id_to_rel[int(labels[b, i])]
                                        for i in range(1, len(heads))]
            predictions[index] = (heads.tolist(), relations)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return predictions, {"seconds": elapsed, "tokens_per_second": sum(map(len, sentences)) / max(elapsed, 1e-9),
                          "sentences_per_second": len(sentences) / max(elapsed, 1e-9)}


def attachment_scores(sentences, predictions):
    if len(sentences) != len(predictions):
        raise ValueError("句子数量不一致")
    totals = {name: [0, 0, 0] for name in ("all_full", "all_coarse", "no_punct_full")}
    for s, (heads, rels) in zip(sentences, predictions):
        gold_heads, gold_rels = s.gold()
        if len(heads) != len(gold_heads) or len(rels) != len(heads):
            raise ValueError("词数不一致")
        for i, row in enumerate(s.tokens, 1):
            arc = int(heads[i] == gold_heads[i])
            full = int(bool(arc) and rels[i] == gold_rels[i])
            coarse = int(bool(arc) and rels[i].split(":")[0] == gold_rels[i].split(":")[0])
            for name, label in (("all_full", full), ("all_coarse", coarse), ("no_punct_full", full)):
                if name == "no_punct_full" and row[3] == "PUNCT":
                    continue
                totals[name][0] += 1
                totals[name][1] += arc
                totals[name][2] += label
    return {name: {"tokens": n, "UAS": 100 * a / max(n, 1), "LAS": 100 * l / max(n, 1)}
            for name, (n, a, l) in totals.items()}


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_checkpoint(state, path):
    '仅用完整写入的断点文件替换旧文件，避免中断时破坏上一轮的断点。'
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_parser(checkpoint, device="cpu"):
    # 仅加载来源可信的本地项目断点文件。
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Parser(state["vocabs"], ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, state["vocabs"], state


def train(name, train_sentences, dev_sentences, vocabs, config, model_config, output_dir, resume=False):
    seed_everything(config["seed"])
    torch.set_num_threads(config.get("threads", 4))
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ParserDataset(train_sentences, vocabs)
    sampler = LengthBatches(dataset, config["batch_size"], config["seed"])
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0)
    model = Parser(vocabs, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, start_epoch, best_las, stale, accumulated_time = [], 1, -1.0, 0, 0.0
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["vocabs"] != vocabs or state["model_config"] != asdict(model_config):
            raise ValueError("恢复训练需要相同的模型配置和训练词表")
        for key in ("seed", "batch_size", "learning_rate", "gradient_clip", "relation_weight", "word_dropout"):
            if state["train_config"].get(key) != config.get(key):
                raise ValueError(f"恢复训练时配置不一致： {key}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_las, stale = state["history"], state["best_las"], state["stale"]
        start_epoch = state["epoch"] + 1
        accumulated_time = state["training_seconds"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if device.startswith("cuda") and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng"]["cuda"]])
        sampler.epoch = start_epoch - 1
    start_time = time.perf_counter()
    parameters = sum(p.numel() for p in model.parameters())
    显示名称 = {"baseline": "MLP 基线", "word_only": "仅词形模型", "word_pos": "词形与词性模型", "notebook_demo": "单轮教学模型"}.get(name, name)
    print(f"{显示名称}：设备={device}，参数量={parameters:,}，训练句数={len(dataset)}", flush=True)
    for epoch in range(start_epoch, config["epochs"] + 1):
        if stale >= config["patience"]:
            break
        model.train()
        total_loss, tokens = 0.0, 0
        epoch_start = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            # 随机替换词语，为未知词嵌入提供训练信号，无需预训练词向量。
            # 人工根节点及填充位置不参与随机替换。
            probability = config.get("word_dropout", 0.0)
            if probability:
                replace = (torch.rand(batch["words"].shape, device=device) < probability) & batch["dep_mask"]
                batch["words"] = batch["words"].masked_fill(replace, vocabs["word"]["<UNK>"])
            optimizer.zero_grad(set_to_none=True)
            arcs, features = model(batch)
            loss = model.loss(batch, arcs, features, config.get("relation_weight", 1.0))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"训练损失不是有限值，发生轮次为 {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            n = int(batch["dep_mask"].sum())
            total_loss += loss.item() * n
            tokens += n
        predictions, _ = infer(model, dev_sentences, vocabs, device, config["batch_size"])
        scores = attachment_scores(dev_sentences, predictions)["all_full"]
        row = {"epoch": epoch, "train_loss": total_loss / tokens, "dev_UAS": scores["UAS"],
               "dev_LAS": scores["LAS"], "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        improved = scores["LAS"] > best_las
        best_las = max(best_las, scores["LAS"])
        stale = 0 if improved else stale + 1
        state = {"name": name, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "model_config": asdict(model_config), "train_config": config, "vocabs": vocabs,
                 "epoch": epoch, "best_las": best_las, "stale": stale, "history": history,
                 "parameters": parameters, "training_seconds": accumulated_time + time.perf_counter() - start_time,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None}}
        save_checkpoint(state, last_path)
        if improved:
            save_checkpoint({k: v for k, v in state.items() if k not in ("optimizer", "rng")}, best_path)
        save_json(history, output_dir / "history.json")
        print(f"{显示名称} 第 {epoch:02d} 轮，损失={row['train_loss']:.4f}，验证 UAS={scores['UAS']:.2f} "
              f"验证 LAS={scores['LAS']:.2f} 耗时（秒）={row['seconds']:.1f} {'*' if improved else ''}", flush=True)
    _, _, best = load_parser(best_path, device="cpu")
    return {"name": name, "parameters": parameters,
            "training_seconds": accumulated_time + time.perf_counter() - start_time,
            "best_epoch": best["epoch"], "best_dev_LAS": best["best_las"],
            "epochs_run": len(history), "device": device}


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        self.model, self.vocabs, _ = load_parser(checkpoint, device)
        self.device = device

    def predict(self, input_conllu_path, output_conllu_path):
        sentences = read_conllu(input_conllu_path)
        predictions, speed = infer(self.model, sentences, self.vocabs, self.device)
        write_predictions(sentences, predictions, output_conllu_path)
        # 重新解析实际写出的文件，验证输出文件中的依存树。
        for sentence in read_conllu(output_conllu_path):
            validate_tree(sentence.gold()[0])
        return predictions, speed
