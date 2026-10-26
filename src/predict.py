'命令行推理入口：读取已分词的 CoNLL-U 文件，预测基本依存树。'
import argparse
from .engine import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--checkpoint", default="checkpoints/full/word_pos/best.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _, speed = Predictor(args.checkpoint, args.device).predict(args.input, args.output)
    print(speed)
