'运行三组实验；在读取测试标签前固定验证集选出的模型。'
import argparse
import hashlib
import importlib.util
import json
import os
import platform
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .data import build_vocabs, dataset_statistics, read_conllu, write_predictions
from .engine import attachment_scores, infer, load_parser, save_json, train
from .model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [("baseline", "baseline", True), ("word_only", "biaffine", False),
               ("word_pos", "biaffine", True)]


def official_scores(gold_path, predicted_path):
    spec = importlib.util.spec_from_file_location("conll18_ud_eval", ROOT / "vendor/conll18_ud_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gold = module.load_conllu_file(str(gold_path))
    predicted = module.load_conllu_file(str(predicted_path))
    scores = module.evaluate(gold, predicted)
    return {key: 100 * scores[key].f1 for key in ("UAS", "LAS")}


def verify_data():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for relative, info in manifest["files"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"数据或评估脚本的文件摘要不一致： {relative}")
    return manifest


def _train_experiment(arguments):
    name, kind, use_pos, sentences, dev, vocabs, config, run_name, resume = arguments
    model_config = ModelConfig(**config["model"], kind=kind, use_pos=use_pos)
    return train(name, sentences, dev, vocabs, config, model_config,
                 ROOT / "checkpoints" / run_name / name, resume)


def run(config_path, resume=False, parallel=False):
    config = json.loads(Path(config_path).read_text())
    manifest = verify_data()
    run_name = config.get("run_name", "full")
    result_dir = ROOT / "experiments" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    splits = {name: read_conllu(ROOT / f"data/raw/en_ewt-ud-{name}.conllu") for name in ("train", "dev")}
    for name in ("train", "dev"):
        if config.get(f"{name}_limit"):
            splits[name] = splits[name][:config[f"{name}_limit"]]
    vocabs = build_vocabs(splits["train"])
    save_json(vocabs, result_dir / "vocabs.json")
    save_json({"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
               "numpy": np.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "dependencies": {name: version(name) for name in ("torch", "numpy", "pandas", "matplotlib", "pytest")},
               "data_commit": manifest["commit"], "parallel_training": parallel,
               "config": config}, result_dir / "environment.json")
    training = []
    arguments = [(name, kind, use_pos, splits["train"], splits["dev"], vocabs, config, run_name, resume)
                 for name, kind, use_pos in EXPERIMENTS]
    if parallel:
        # 各独立进程分别维护随机数状态和训练断点。
        # 等待所有训练进程退出后再评估，避免推理测速受到训练任务干扰。
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
            training = list(pool.map(_train_experiment, arguments))
        save_json(training, result_dir / "training_summary.json")
    else:
        for args in arguments:
            training.append(_train_experiment(args))
            save_json(training, result_dir / "training_summary.json")
    # 在加载测试集之前，仅依据验证集 LAS 确定最终模型。
    selected = max(training, key=lambda row: row["best_dev_LAS"])["name"]
    save_json({"selected_model": selected, "criterion": "maximum validation all-token full-label LAS",
               "test_used_for_selection": False}, result_dir / "selection.json")
    splits["test"] = read_conllu(ROOT / "data/raw/en_ewt-ud-test.conllu")
    save_json(dataset_statistics(splits, vocabs), result_dir / "dataset_statistics.json")
    results = []
    for record in training:
        name = record["name"]
        model, model_vocabs, _ = load_parser(ROOT / "checkpoints" / run_name / name / "best.pt", record["device"])
        predictions, speed = infer(model, splits["test"], model_vocabs, record["device"], config["batch_size"])
        prediction_path = result_dir / f"{name}_test.conllu"
        write_predictions(splits["test"], predictions, prediction_path)
        reparsed = read_conllu(prediction_path)
        for s in reparsed:
            from .decoder import validate_tree
            validate_tree(s.gold()[0])
        custom = attachment_scores(splits["test"], predictions)
        official = official_scores(ROOT / "data/raw/en_ewt-ud-test.conllu", prediction_path)
        for key in ("UAS", "LAS"):
            if abs(custom["all_coarse"][key] - official[key]) > 1e-8:
                raise AssertionError(f"官方与自定义指标不一致，模型为 {name}: {key}")
        row = {**record, "test": custom, "official": official, "inference": speed}
        results.append(row)
        save_json(results, result_dir / "results.json")
        print(f"测试集结果 {name}： {custom['all_full']} 官方得分={official}", flush=True)
    from .report import generate_artifacts
    generate_artifacts(run_name)
    print(f"实验完成：experiments/{run_name}；验证集选出的模型：{selected}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="并行训练三组模型，需要足够的显存")
    args = parser.parse_args()
    run(args.config, args.resume, args.parallel)
