'根据真实预测结果生成可复现的统计表、图表和分析报告。'
from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd

# 优先使用系统中文字体；最后保留通用字体以显示英文和数学符号。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
显示分组 = {"train": "训练集", "dev": "验证集", "test": "测试集",
    "Short (1-10)": "短句（1–10 词）", "Medium (11-25)": "中句（11–25 词）", "Long (26+)": "长句（26 词及以上）"}
报告列名 = {"Model": "模型", "Official LAS": "官方 LAS", "Parameters": "参数量",
    "Training seconds": "训练耗时（秒）", "Tokens/second": "每秒处理词数", "Best epoch": "最佳轮次", "Dev LAS": "验证集 LAS",
    "Tokens": "词数", "distance_group": "依存距离分组", "length_group": "句长分组", "relation": "依存关系"}

def 中文统计表(table):
    """只转换展示标签，不修改原始数据和机器可读字段。"""
    table = table.rename(index=显示分组, columns=报告列名)
    table.index.name = 报告列名.get(table.index.name, table.index.name)
    return table

from .data import read_conllu, write_predictions
from .engine import save_json

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"baseline": "MLP 基线", "word_only": "BiLSTM + 双仿射（仅词形）",
          "word_pos": "BiLSTM + 双仿射（词形与词性）"}


def prediction_pairs(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError("句子数量不一致")
    pairs = []
    for g, p in zip(gold, predicted):
        if [r[1] for r in g.tokens] != [r[1] for r in p.tokens]:
            raise ValueError("词语对齐不一致")
        h, r = p.gold()
        pairs.append((h, r))
    return pairs


def token_records(gold, predictions, vocabs):
    rows = []
    for sentence_id, (s, (heads, relations)) in enumerate(zip(gold, predictions)):
        gh, gr = s.gold()
        for i, row in enumerate(s.tokens, 1):
            distance = abs(i - gh[i])
            rows.append({"sentence": sentence_id, "position": i, "word": row[1], "upos": row[3],
                         "gold_head": gh[i], "pred_head": heads[i], "gold_rel": gr[i], "pred_rel": relations[i],
                         "distance": distance, "distance_group": "1" if distance == 1 else "2-3" if distance <= 3 else "4-6" if distance <= 6 else "7+",
                         "relation": gr[i].split(":")[0], "length": len(s),
                         "length_group": "Short (1-10)" if len(s) <= 10 else "Medium (11-25)" if len(s) <= 25 else "Long (26+)",
                         "oov": row[1].lower() not in vocabs["word"],
                         "arc_correct": int(heads[i] == gh[i]),
                         "label_correct": int(heads[i] == gh[i] and relations[i] == gr[i])})
    return pd.DataFrame(rows)


def grouped_scores(records, key, order=None):
    table = records.groupby(key, observed=True).agg(Tokens=("word", "size"),
                  UAS=("arc_correct", "mean"), LAS=("label_correct", "mean"))
    table[["UAS", "LAS"]] *= 100
    return table.reindex(order).dropna() if order else table.sort_values("Tokens", ascending=False)


def draw_tree(ax, sentence, heads, relations, title, wrong=None):
    '绘制完整的短句依存树，展示全部依存弧及其关系标签。'
    words = ["ROOT"] + [r[1] for r in sentence.tokens]
    n = len(words)
    max_height = 1
    for dep in range(1, n):
        head = int(heads[dep])
        distance = abs(dep - head)
        height = 0.7 + 0.32 * distance
        max_height = max(max_height, height)
        color = "#c44e52" if wrong and dep in wrong else "#326b9b"
        # 使用带箭头的平滑二次曲线绘制依存弧。
        arrow = FancyArrowPatch((head, 0.15), (dep, 0.15),
                 connectionstyle=f"arc3,rad={-0.20 if head < dep else 0.20}",
                 arrowstyle="-|>", mutation_scale=10, color=color, linewidth=1.2)
        ax.add_patch(arrow)
        arc_y = 0.15 + distance * 0.10
        ax.text((head + dep) / 2, arc_y + 0.08, relations[dep], ha="center", fontsize=7,
                color=color, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2})
    for i, word in enumerate(words):
        ax.text(i, 0, word, ha="center", va="top", fontsize=8)
        ax.text(i, -0.24, str(i), ha="center", fontsize=7, color="gray")
    ax.set(xlim=(-0.6, n - 0.4), ylim=(-0.55, max(1.5, n * 0.13)), title=title)
    ax.axis("off")


def generate_artifacts(run_name="full"):
    result_dir = ROOT / "experiments" / run_name
    figures = result_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = json.loads((result_dir / "results.json").read_text())
    selection = json.loads((result_dir / "selection.json").read_text())
    selected = selection["selected_model"]
    vocabs = json.loads((result_dir / "vocabs.json").read_text())
    splits = {s: read_conllu(ROOT / f"data/raw/en_ewt-ud-{s}.conllu") for s in ("train", "dev", "test")}
    gold = splits["test"]
    predicted = read_conllu(result_dir / f"{selected}_test.conllu")
    predictions = prediction_pairs(gold, predicted)
    records = token_records(gold, predictions, vocabs)
    records.to_csv(result_dir / "token_errors.csv", index=False)
    comparison = pd.DataFrame([{"Model": LABELS[r["name"]], "UAS": r["test"]["all_full"]["UAS"],
                    "LAS": r["test"]["all_full"]["LAS"], "Official LAS": r["official"]["LAS"],
                    "Parameters": r["parameters"], "Training seconds": r["training_seconds"],
                    "Tokens/second": r["inference"]["tokens_per_second"],
                    "Best epoch": r["best_epoch"], "Dev LAS": r["best_dev_LAS"]} for r in results])
    comparison.to_csv(result_dir / "comparison.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    comparison.set_index("Model")[["UAS", "LAS"]].plot.bar(ax=ax, color=["#3b83a7", "#dd9858"], rot=10)
    ax.set(title="测试集依存分析得分（包括标点，比较完整标签）", ylabel="准确率（%）", ylim=(0, 100))
    ax.set_xlabel("")
    save(fig, "model_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for row in results:
        history = pd.read_json(ROOT / "checkpoints" / run_name / row["name"] / "history.json")
        for ax, key in zip(axes, ("train_loss", "dev_UAS", "dev_LAS")):
            ax.plot(history.epoch, history[key], label=LABELS[row["name"]])
    for ax, title, ylabel in zip(axes, ("训练损失", "验证集 UAS", "验证集 LAS"), ("损失", "UAS (%)", "LAS (%)")):
        ax.set(title=title, xlabel="训练轮次", ylabel=ylabel)
    axes[0].legend(fontsize=7)
    save(fig, "training_curves")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    groups = [("distance_group", ["1", "2-3", "4-6", "7+"], "金标准依存距离"),
              ("length_group", ["Short (1-10)", "Medium (11-25)", "Long (26+)"], "句子长度"),
              ("relation", None, "高频金标准依存关系")]
    tables = {}
    for ax, (key, order, title) in zip(axes, groups):
        table = grouped_scores(records, key, order)
        tables[key] = table.reset_index().to_dict("records")
        table.to_csv(result_dir / f"errors_by_{key}.csv")
        (table if key != "relation" else table.head(12)).rename(index=显示分组)[["UAS", "LAS"]].plot.bar(
            ax=ax, color=["#3b83a7", "#dd9858"], rot=35)
        ax.set(title=title, ylabel="准确率（%）", xlabel="", ylim=(0, 100))
        ax.legend(fontsize=7)
    save_json(tables, result_dir / "error_groups.json")
    save(fig, "error_analysis")
    grouped_scores(records, "oov").to_csv(result_dir / "errors_by_oov.csv")

    # 此处仅作探索性统计，词表与特征选择仍只使用训练数据。
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for name, sentences in splits.items():
        axes[0].hist([len(s) for s in sentences], bins=range(0, 101, 5), alpha=0.45, label=显示分组.get(name, name))
    axes[0].set(title="句子长度分布（显示至 100 词）", xlabel="句法词数", ylabel="句子数量")
    axes[0].legend()
    for ax, column, title in ((axes[1], 3, "训练集通用词性分布"), (axes[2], 7, "训练集高频依存关系（前 15 类）")):
        counter = Counter(r[column] for s in splits["train"] for r in s.tokens)
        pd.Series(dict(counter.most_common(15))).plot.bar(ax=ax, rot=50, color="#3b83a7")
        ax.set(title=title, ylabel="词数", xlabel="")
    save(fig, "data_exploration")
    for column, name in ((3, "pos_frequencies"), (7, "relation_frequencies")):
        counter = Counter(r[column] for s in splits["train"] for r in s.tokens)
        pd.DataFrame(counter.most_common(), columns=["label", "tokens"]).to_csv(result_dir / f"{name}.csv", index=False)

    cases = []
    for index, (s, (heads, rels)) in enumerate(zip(gold, predictions)):
        gh, gr = s.gold()
        wrong = [i for i in range(1, len(gh)) if gh[i] != heads[i] or gr[i] != rels[i]]
        if 4 <= len(s) <= 12 and wrong:
            cases.append({"index": index, "text": s.text, "gold_heads": gh, "gold_relations": gr,
                          "pred_heads": heads, "pred_relations": rels, "wrong_positions": wrong})
        if len(cases) == 3:
            break
    save_json(cases, result_dir / "examples.json")
    write_predictions([gold[c["index"]] for c in cases],
                      [(c["pred_heads"], c["pred_relations"]) for c in cases], result_dir / "examples.conllu")
    for number, case in enumerate(cases, 1):
        fig, axes = plt.subplots(2, 1, figsize=(12, 5))
        sentence = gold[case["index"]]
        draw_tree(axes[0], sentence, case["gold_heads"], case["gold_relations"], "金标准依存树")
        draw_tree(axes[1], sentence, case["pred_heads"], case["pred_relations"],
                  "预测依存树（红色表示父节点或关系错误）", set(case["wrong_positions"]))
        save(fig, f"example_{number}")

    word = next(r for r in results if r["name"] == "word_only")
    pos = next(r for r in results if r["name"] == "word_pos")
    chosen = next(r for r in results if r["name"] == selected)
    worst = grouped_scores(records, "relation")
    worst = worst[worst.Tokens >= 100].sort_values("LAS").head(5)
    headers = [报告列名.get(column, column) for column in comparison.columns]
    markdown_table = "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(f"{value:.2f}" if isinstance(value, float) else str(value) for value in row) + " |"
          for row in comparison.itertuples(index=False, name=None)]])
    text = ["# 真实实验结果", "", f"仅依据验证集 LAS 选出的模型：**{LABELS[selected]}**。", "",
            "测试使用金标准分词；词性模型使用金标准 UPOS。所有词嵌入均随机初始化。", "",
            "主要指标包括标点，完整比较依存关系标签及其子类型。",
            "官方指标包括标点，但去除关系子类型；自定义粗粒度得分与官方脚本完全一致。", "",
            markdown_table, "",
            f"所选模型的测试 UAS 为 {chosen['test']['all_full']['UAS']:.2f}%，LAS 为 {chosen['test']['all_full']['LAS']:.2f}%。",
            f"加入词性后的测试 LAS 相对仅词形模型变化 {pos['test']['all_full']['LAS'] - word['test']['all_full']['LAS']:+.2f} 个百分点。", "",
            "## 所选模型的分组错误分析", ""]
    for key in ("distance_group", "length_group"):
        text.extend(["```text", 中文统计表(grouped_scores(records, key, dict((k, o) for k, o, _ in groups)[key])).to_string(float_format=lambda x: f"{x:.2f}"), "```", ""])
    text.extend(["测试集出现次数不少于 100 的关系中，LAS 最低的类别：", "```text", 中文统计表(worst).to_string(float_format=lambda x: f"{x:.2f}"), "```", "",
                 "这些结果描述单个随机种子下的现象，不构成因果关系的统计证据。依存距离分组包括根弧。",
                 "未登录词使用训练得到的 UNK 向量。错误分组可能重叠，相关性不能确定唯一错误原因。", "",
                 "局限性：采用金标准分词与词性，仅覆盖一个树库和一个随机种子，未使用字符模型，也未评估预测词性输入。",
                 "后续改进：加入字符编码、预测词性、多随机种子实验，以及仅在验证集上进行的独立调参。", "",
                 "三组模型均在最终完整测试集评估之前，根据验证集 LAS 选择最佳断点。",
                 "训练墙钟时间可能包含共享 GPU 的资源竞争；是否并行训练记录在 environment.json 的 parallel_training 字段中。",
                 "推理速度包括神经评分、CPU 上的 MST 解码及关系预测，不包括磁盘读写和输入编码。"])
    (result_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return comparison, records
