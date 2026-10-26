# 从零实现依存句法分析器

一个适合课程演示的小型 PyTorch 项目：**Word/POS Embedding → BiLSTM → Biaffine → 单根 MST 解码**。全部参数随机初始化，不使用预训练词向量、语言模型或现成依存分析器。

优先打开 [逐步演示 Notebook](notebooks/演示.ipynb)。Notebook 使用中文，依次解释数据、特征、模型、树解码、训练、评估和错误分析；已保留真实运行输出。姓名、学号留有待填写位置。

已实际通过 18 项测试；Notebook 的 17 个代码单元已从干净内核顺序执行成功，无错误输出。交付检查记录及 Notebook/最佳权重的 SHA-256 保存在 `experiments/full/verification.json`。

项目自有代码的注释、文档字符串、Notebook 说明、展示表格和图表均使用中文。英文语料、UD 标准标签、机器可读字段名，以及官方评估脚本的原始内容和版权声明保留原样。

## 快速开始

在项目根目录运行（Python 3.10+）：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
jupyter notebook notebooks/演示.ipynb
```

也可以用 VS Code 的 Jupyter 扩展直接打开 Notebook。默认读取随项目交付的完整实验结果，并额外运行一次很短的 CPU 教学训练，无需重新完成全部训练。

## 重新训练

数据已经随项目保存，严格采用 **UD 2.15 English-EWT**，固定提交为 `4dc8e10cf32352e11ab2c46e024b19853b91546e`。`data/manifest.json` 保存来源、文件大小和 SHA-256；训练前会检查所有文件的摘要。

训练、验证、测试集分别为 12,544 / 2,001 / 2,077 句，204,579 / 25,149 / 25,094 个句法词。小写词形表含 16,657 项（含三个特殊符号），验证/测试词级 OOV 比例约为 6.80% / 7.50%。

```powershell
python scripts/download_data.py
python -m src.experiment --config configs/full.json
```

完整命令依次完成三组实验：

1. **baseline**：Word + POS、MLP 编码和弧/关系评分，无 LSTM、无 Biaffine。
2. **word_only**：Word、BiLSTM、弧/关系 Biaffine。
3. **word_pos**：Word + POS、BiLSTM、弧/关系 Biaffine。

默认使用题目给定的维度、Adam、学习率 0.001、batch size 32、最多 30 epoch、patience 5、梯度裁剪 5、随机种子 42；另使用 0.10 的随机词替换训练 UNK 表示。配置集中在 `configs/full.json`。自动选择 CUDA/CPU，按近似长度组批。三组实验保持相同训练预算和选择规则。每个模型只依据验证集的完整标签 LAS 保存最佳权重；三组模型全部完成后，先冻结验证集选出的模型，再评估完整测试集。

显存足够时可加 `--parallel`，用三个独立进程同时训练。`environment.json` 记录是否并行；此时训练耗时是共享 GPU 下的墙钟时间，不能当成独占 GPU 的速度基准。最终推理测速在全部训练完成后逐模型进行。

断点从已完成的 epoch 边界恢复，包含优化器和随机数状态：

```powershell
python -m src.experiment --config configs/full.json --resume
```

如只想验证流程，可执行以下小规模命令。其输出位于独立的 `demo` 目录，**不能作为完整训练结果**：

```powershell
python -m src.experiment --config configs/demo.json
```

## 使用已训练模型

```powershell
python -m src.predict input.conllu output.conllu --checkpoint checkpoints/full/word_pos/best.pt
```

或在 Python 中调用：

```python
from src.engine import Predictor
predictor = Predictor("checkpoints/full/word_pos/best.pt")
predictor.predict("input.conllu", "output.conllu")
```

输入需要已有的分词及 UPOS；Word-only 模型不使用 POS。HEAD/DEPREL 可以是 `_`，推理数据集不会读取这些字段。输出保留注释、多词 token 和表层字段，清空过期的 DEPS，并删除仅服务于增强依存图的空节点。每个词有一个父节点，每句有且只有一个根词，无环、无自环，`HEAD=0` 与 `DEPREL=root` 一致。代码会重新读取输出并验证树。

## 结果与口径

完整测试集的实测结果如下（包括标点，LAS 完整比较关系子类型；POS 组使用金标准词性）：

| 模型 | UAS (%) | LAS (%) |
| --- | ---: | ---: |
| MLP 基线 | 38.12 | 33.13 |
| BiLSTM + Biaffine（Word） | 84.02 | 79.23 |
| BiLSTM + Biaffine（Word + POS） | **89.69** | **87.83** |

验证集选出的最佳模型为 Word + POS；相对 Word-only 的测试 LAS 增加 **8.60 个百分点**。其官方粗粒度 LAS 为 **88.03%**。全部模型均完成 30 轮训练，具体最佳轮次、耗时和参数量以结果文件为准。

完整结果位于 `experiments/full/`：

- `comparison.csv`、`results.json`：三组实测 UAS/LAS、参数量、训练耗时、推理速度。
- `report.md`：从实测数据生成的对比和局限性说明。
- `selection.json`：在查看测试标签前确定的验证集选择结果。
- `environment.json`：实际训练的软件版本、GPU 和配置。
- `*_test.conllu`：三组模型对完整测试集的预测。
- `token_errors.csv`、`errors_by_*.csv`：按距离、关系、句长、OOV 的错误统计。
- `examples.conllu`、`examples.json`：三个完整实例。
- `figures/`：训练曲线、对比图、错误分析及 gold/predicted 树。

主表的自定义指标**包括标点，完整比较关系子类型**。同时给出包括标点、去除 `:subtype` 的粗粒度得分，并与官方 CoNLL-2018 脚本逐项核对。另提供按 gold UPOS=`PUNCT` 排除标点的完整标签得分。不要混用不同口径的 LAS。

**实验假设：使用 gold tokenization；POS 组使用 gold UPOS。** 这不是原始文本自动分词/词性标注系统。全部词表只由训练集建立，ROOT、padding、多词行、空节点不进入评分分母。依存距离按 gold head 的内部位置计算，包括根弧。推理速度包括神经评分、CPU 解码和关系预测，不包括磁盘读写和输入编码；训练时间包含每轮验证。

结果仅基于一个语料、一个随机种子，不能据此作统计显著性结论。错误分组可能重叠；错误原因只能作为有待进一步验证的解释。不同硬件和 PyTorch/CUDA 版本不保证逐位相同。

## 代码位置

```text
src/data.py          CoNLL-U、词表、Dataset、Padding/Mask
src/model.py         MLP 基线、BiLSTM、Biaffine、联合损失
src/decoder.py       自行实现的 Chu–Liu/Edmonds 和精确单根处理
src/engine.py        训练、早停、断点、指标、推理接口
src/experiment.py    三组实验、验证集选择、官方交叉检查
src/report.py        CSV、图表、错误实例和结果报告
src/predict.py       CoNLL-U 推理命令
tests/              解码穷举、数据、模型与断点恢复测试
checkpoints/full/   每组 best.pt、last.pt、history.json
notebooks/          已执行的逐步演示 Notebook
```

解码器先解不限制根子节点数量的最大有向生成树，若已单根则直接返回；否则对根词候选显式约束并求解，使用安全上界剪枝。测试对小图穷举所有合法单根树，覆盖单词、多根、环、非投射、并列得分和缺失边。

## 数据与第三方脚本

- [UD English-EWT r2.15](https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15)，原始许可证保存在 `data/raw/LICENSE.txt`，保留语料 README。
- [官方评估规则与脚本](https://universaldependencies.org/conll18/evaluation.html)，未修改的脚本保存在 `vendor/conll18_ud_eval.py`，其版权及 MPL-2.0 声明保留在文件头部。
- [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)。GPU 完整训练环境会在 `environment.json` 中记录，Notebook 可在 CPU 上查看和演示。

项目文件不包含服务器登录密码。查看 Notebook 和使用已训练权重不需要连接训练服务器。
