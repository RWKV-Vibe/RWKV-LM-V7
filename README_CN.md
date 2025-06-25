
<div align="center">

# RWKV-LM-V7
[![English](https://img.shields.io/badge/README-English-blue.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-中文版本-red.svg)](./README_CN.md)

</div>

## 项目介绍
让任何研究者在 15 分钟内开始预训练一个完全对齐的 RWKV v7 模型。当然不包括下载数据 :) 。

所有的代码来源于原始 RWKV-LM 项目：https://github.com/BlinkDL/RWKV-LM

此仓库适合快速的在英伟达显卡上使用样例数据或私有数据小规模的复现 RWKV v7 系列模型，例如 191M ~ 3B 等大小，我们接下来会重点改善：

- 提供 RWKV 系列模型在多模态等任务的模板代码
- 提供跨平台的内核实现
- 提供可配置的 RWKV Layer 类
- 提供高性能的 Pytorch 推理实现
- 提供适合 3 ~ 70B 的集群训练框架及脚本

我们热爱并回馈开源社区，感谢任何开源社区的实现。如果您发现我们的代码仓库有包含但不限于：代码质量，代码风格，代码可解释性，数值精度误差的问题，欢迎[提交 issue](https://github.com/RWKV-Vibe/RWKV-LM-V7/issues/new)。

> [!WARNING]
> Note: 整个仓库仍处于 WIP 阶段（与基线相比，我们改进了融合算子的使用，计算更高效）。如果您有所顾虑，则可以使用 [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp) 作为参考实现。

## 如何开始？

### 准备环境

环境准备，请使用 miniforge 等 conda 兼容包管理器，创建一个全新的环境：
```
conda create -n rwkv-lm-v7 python=3.12
conda activate rwkv-lm-v7
```
随后安装下列依赖，注意 `pytorch-lightning` 固定使用了 `1.9.5` 版本，此为本仓库特性，请不要升级此依赖包。
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install -r requirements.txt
```

### 下载数据

```
wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

### 开始训练

1. 初始化空 RWKV-7 模型
```
sh ./demo-training-prepare.sh
```

2. 登录 WandB 账号

3. 开始训练
```
sh ./demo-training-run.sh
```

## 详细解释

此章节包含模型初始化、学习率及细节解释。

RWKV-7 使用了包含经过设计和数学论证的初始化和基于训练结果分析的初始化，加速模型收敛及其性能。

### L2Warp
此类惩罚模型，避免模型过度自信，从而缓解 BF16 中间的精度损失。

### 权重及其初始化样例

请严格注意上下文学习率等相关设置：
```
self.k_k = nn.Parameter(torch.zeros(1, 1, C)+0.71 - linear*0.1)
self.k_a = nn.Parameter(torch.zeros(1, 1, C)+1.02)
```

RWKV-7 weight example for 1.5B (L24-D2048, vocab 65536):

| name                | shape         | comment      | initialization  |
|---------------------|---------------|--------------|-----------------|
| emb.weight          | [65536, 2048] | wdecay       | see code        |
| blocks.0.ln0.weight | [2048]        | for layer 0  | 1               |
| blocks.0.ln0.bias   | [2048]        | for layer 0  | 0               |
|                     |               |              |                 |
| blocks.*.ln1.weight | [2048]        |              | 1               |
| blocks.*.ln1.bias   | [2048]        |              | 0               |
| blocks.*.att.x_r    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_w    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_k    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_v    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_a    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_g    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.w0     | [1, 1, 2048]  | lr 2x        | see code        |
| blocks.*.att.w1     | [2048, 96]    |              | 0               |
| blocks.*.att.w2     | [96, 2048]    |              | see code        |
| blocks.*.att.a0     | [1, 1, 2048]  |              | 0               |
| blocks.*.att.a1     | [2048, 96]    |              | 0               |
| blocks.*.att.a2     | [96, 2048]    |              | see code        |
| blocks.*.att.v0     | [1, 1, 2048]  | for layer 1+ | 1               |
| blocks.*.att.v1                | [2048, 64]   | for layer 1+ | 0         |
| blocks.*.att.v2                | [64, 2048]   | for layer 1+ | see code  |
| blocks.*.att.g1                | [2048, 256]  |              | 0         |
| blocks.*.att.g2                | [256, 2048]  |              | see code  |
| blocks.*.att.k_k               | [1, 1, 2048] |              | 1         |
| blocks.*.att.k_a               | [1, 1, 2048] |              | 1         |
| blocks.*.att.r_k               | [32, 64]     |              | 0         |
| blocks.*.att.receptance.weight | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.key.weight        | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.value.weight      | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.output.weight     | [2048, 2048] | wdecay       | 0         |
| blocks.*.att.ln_x.weight       | [2048]       |              | see code  |
| blocks.*.att.ln_x.bias         | [2048]       |              | 0         |
|                                |              |              |           |
| blocks.*.ln2.weight            | [2048]       |              | 1         |
| blocks.*.ln2.bias              | [2048]       |              | 0         |
| blocks.*.ffn.x_k               | [1, 1, 2048] |              | see code  |
| blocks.*.ffn.key.weight        | [8192, 2048] | wdecay       | see code  |
| blocks.*.ffn.value.weight      | [2048, 8192] | wdecay       | 0         |
|                                |              |              |           |
| ln_out.weight | [2048]        |        | 1         |
| ln_out.bias   | [2048]        |        | 0         |
| head.weight   | [65536, 2048] | wdecay | see code  |

## 检查结果

在 `out/....../train_log.txt` 路径下，您的损失应该非常接近：

```
0 4.875856 131.0863 0.00059975 2025-04-24 02:23:42.481256 0
1 4.028621 56.1834 0.00059899 2025-04-24 02:28:16.674463 1
2 3.801625 44.7739 0.00059773 2025-04-24 02:32:51.059568 2
3 3.663070 38.9808 0.00059597 2025-04-24 02:37:25.409892 3
4 3.578974 35.8368 0.00059371 2025-04-24 02:41:59.711315 4
5 3.510906 33.4786 0.00059096 2025-04-24 02:46:33.990839 5
6 3.462345 31.8917 0.00058771 2025-04-24 02:51:08.378331 6
7 3.412196 30.3318 0.00058399 2025-04-24 02:55:42.927474 7
8 3.376724 29.2747 0.00057978 2025-04-24 03:00:17.504665 8
9 3.336911 28.1321 0.00057511 2025-04-24 03:04:52.006063 9
10 3.313411 27.4787 0.00056999 2025-04-24 03:09:27.563336 10
11 3.295895 27.0016 0.00056441 2025-04-24 03:14:01.786079 11
```

## 处理训练数据

### 将 jsonl 转换为 binidx 格式

使用 `data/make_data.py` 脚本将你的训练数据从 `.jsonl` 格式转换为 `binidx` 格式。

```
python make_data.py [输入文件] [重复轮数] [上下文长度]
# 示例：
cd data/
python make_data.py demo.jsonl 3 4096
```

`python make_data.py demo.jsonl 3 4096` 命令将会：

- 对 demo.jsonl 进行打乱并复制 3 轮
- 加载 jsonl 并进行 tokenize
- 保存为 demo.bin 和 demo.idx
- 计算 ctxlen=4096 时的 `magic_prime`

假设你的源 jsonl 如下：

- {"text":"aa"}
- {"text":"bb"}
- {"text":"cc"}
- {"text":"dd"}

最终的 binidx 会像这样（这里 "/" 表示 end_of_doc，实际是 token [0]）：`bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/`

> [!WARNING]
> make_data.py 处理大体积 jsonl 时会非常慢，如需处理大文件请参考 [json2binidx_tool](https://github.com/Abel2076/json2binidx_tool)。

### 为指定 binidx 数据集计算 magic_prime

`data/compute_magic_prime.py` 脚本可为指定的 binidx 数据集和上下文长度（ctx_len）计算正确的 `--my_exit_tokens` 和 `--magic_prime` 值。

1. 在 `data/compute_magic_prime.py` 中修改你的训练数据集和上下文长度（`DATA_NAME` 和 `CTX_LEN`）
2. 运行脚本以获得正确的 `--my_exit_tokens` 和 `--magic_prime` 值

```
cd data/
python compute_magic_prime.py
```

最终输出类似于：

```
### Loading /home/rwkv/RWKV-LM-V7/data/demo

### /home/rwkv/RWKV-LM-V7/data/demo.bin/idx has 200499 tokens, 546 items. Dtype <class 'numpy.uint16'>

### magic_prime = 47 (for ctxlen 4096)

--my_exit_tokens 200499 --magic_prime 47 --ctx_len 4096
```
