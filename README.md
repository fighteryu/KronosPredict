# Kronos Qlib K-Line Predictor

基于本地 `Kronos` 模型和 `qlib` 数据，在 **CPU** 上预测未来 **交易日** K 线走势，并输出：

- 预测结果 CSV：`kronos_pred.csv`
- K 线图 PNG：`kronos_pred.png`（单图，上预测下真实）

脚本文件：

- `kronos_qlib_predict.py`

## 功能说明

脚本支持两类模型输入：

1. `Kronos` 官方模型目录
   - 目录中包含 `config.json`
   - 目录中包含 `*.safetensors`
   - 通过官方 `Kronos + KronosTokenizer + KronosPredictor` 推理

2. 普通 PyTorch 模型文件
   - 如 `*.pt`、`*.pth`、`*.ckpt`、`*.bin`
   - 通过 `torch.load(..., map_location="cpu")` 加载

默认目标是：

- 从 `qlib` 读取单只股票日线 OHLCV
- 用历史窗口构造输入
- 预测未来 `5` 个交易日
- 生成单张 K 线图（上预测，下真实；无真实数据时仅显示预测）

## 目录建议

推荐目录结构如下：

```text
kronos_demo/
├── kronos_qlib_predict.py
├── README.md
├── qlib_data/
├── model/
├── tokenizer/
└── Kronos/
```

说明：

- `qlib_data/`：本地 qlib 数据目录
- `model/`：本地 Kronos 模型目录
- `tokenizer/`：本地 Kronos tokenizer 目录
- `Kronos/`：官方源码仓库，用于提供 `model.py`

## 环境要求

- Python 3.10+
- CPU 环境即可
- 已安装本地 `qlib` 数据

建议依赖：

```bash
python -m pip install -U pyqlib pandas numpy matplotlib safetensors
python -m pip install torch==2.2.2
```

如果你使用官方 `Kronos` 源码中的 `requirements.txt`，也可以直接：

```bash
cd ~/Downloads/kronos_demo/Kronos
python -m pip install -r requirements.txt
```

## 下载模型和 tokenizer

### 1. 下载 Kronos 模型

推荐使用 Hugging Face Hub 的整仓下载，而不是手动拷贝单个文件：

```bash
python -m pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='NeoQuasar/Kronos-base', local_dir='/Users/fighteryu/Downloads/kronos_demo/model', local_dir_use_symlinks=False)"
```

### 2. 下载 tokenizer

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='NeoQuasar/Kronos-Tokenizer-base', local_dir='/Users/fighteryu/Downloads/kronos_demo/tokenizer', local_dir_use_symlinks=False)"
```

### 3. 检查下载结果

模型目录中至少应包含：

- `config.json`
- `model.safetensors` 或其他 `*.safetensors`

tokenizer 目录中应包含 tokenizer 所需配置和词表文件。

可用下面命令检查：

```bash
ls -la ~/Downloads/kronos_demo/model
ls -la ~/Downloads/kronos_demo/tokenizer
```

## 下载官方 Kronos 源码

当前脚本在加载 `Kronos` 官方模型时，会使用：

```python
from model import Kronos, KronosTokenizer, KronosPredictor
```

因此需要本地存在官方代码仓库：

```bash
cd ~/Downloads/kronos_demo
git clone https://github.com/shiyu-coder/Kronos.git
```

运行脚本时，需要把 `Kronos` 仓库加入 `PYTHONPATH`：

```bash
PYTHONPATH=~/Downloads/kronos_demo/Kronos python ~/Downloads/kronos_demo/kronos_qlib_predict.py ...
```

## 使用方法

### 标准运行命令

```bash
PYTHONPATH=~/Downloads/kronos_demo/Kronos python ~/Downloads/kronos_demo/kronos_qlib_predict.py \
  --provider-uri ~/Downloads/kronos_demo/qlib_data \
  --instrument sh600519 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --model-path ~/Downloads/kronos_demo/model \
  --tokenizer-path ~/Downloads/kronos_demo/tokenizer \
  --window 64 \
  --horizon 5 \
  --seed 40 \
  --out ~/Downloads/kronos_demo/kronos_pred.csv \
  --chart-out ~/Downloads/kronos_demo/kronos_pred.png
```

### 离线运行

如果模型和 tokenizer 都已经完整下载到本地，可以强制离线：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=~/Downloads/kronos_demo/Kronos python ~/Downloads/kronos_demo/kronos_qlib_predict.py \
  --provider-uri ~/Downloads/kronos_demo/qlib_data \
  --instrument sh600519 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --model-path ~/Downloads/kronos_demo/model \
  --tokenizer-path ~/Downloads/kronos_demo/tokenizer \
  --window 64 \
  --horizon 5 \
  --seed 40 \
  --out ~/Downloads/kronos_demo/kronos_pred.csv \
  --chart-out ~/Downloads/kronos_demo/kronos_pred.png
```

### 参数搜索（自动回测）

如果你要自动找更优参数（`window / T / top_p / sample_count`），可以使用 `--tune`：

```bash
PYTHONPATH=~/Downloads/kronos_demo/Kronos python ~/Downloads/kronos_demo/kronos_qlib_predict.py \
  --provider-uri ~/Downloads/kronos_demo/qlib_data \
  --instrument sh600519 \
  --start 2021-01-01 \
  --end 2024-12-31 \
  --model-path ~/Downloads/kronos_demo/model \
  --tokenizer-path ~/Downloads/kronos_demo/tokenizer \
  --horizon 5 \
  --seed 40 \
  --tune \
  --grid-window 64,128,256,384 \
  --grid-temp 1.0,0.9,0.7 \
  --grid-top-p 0.95,0.9,0.8 \
  --grid-sample-count 1,5,10 \
  --tune-stride 5 \
  --tune-max-windows 120 \
  --tune-out ~/Downloads/kronos_demo/kronos_tune_scores.csv
```

说明：

- 会在历史区间做滚动回测
- 评分指标为 `close` 的 `MAE / RMSE / MAPE`
- 结果会保存到 `--tune-out`
- 程序会打印 `RMSE(close)` 最优参数组合

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `--provider-uri` | qlib 数据目录 |
| `--region` | `cn` 或 `us` |
| `--instrument` | 单个标的，如 `sh600519` |
| `--start` | 历史数据开始日期 |
| `--end` | 历史数据结束日期 |
| `--model-path` | 本地模型目录或模型文件 |
| `--tokenizer-path` | 本地 tokenizer 目录或 Hugging Face repo id |
| `--window` | 输入历史窗口长度 |
| `--horizon` | 预测未来交易日数量，默认 `10` |
| `--batch-size` | 普通 PyTorch 模型批量推理大小 |
| `--seed` | 随机种子，默认 `40` |
| `--out` | 预测 CSV 输出路径 |
| `--chart-out` | 统一K线图 PNG 输出路径（上预测下真实） |
| `--tune` | 开启自动回测参数搜索模式 |
| `--tune-out` | 参数搜索评分结果 CSV 路径 |
| `--tune-stride` | 滚动回测步长（bar 数） |
| `--tune-max-windows` | 最多评估的滚动窗口数量 |
| `--grid-window` | 窗口候选列表（逗号分隔） |
| `--grid-temp` | 温度候选列表（逗号分隔） |
| `--grid-top-p` | top-p 候选列表（逗号分隔） |
| `--grid-sample-count` | sample_count 候选列表（逗号分隔） |

## 输出说明

### 1. CSV 输出

输出文件默认是 `kronos_pred.csv`，包含类似字段：

- `date`
- `symbol`
- `pred_open`
- `pred_high`
- `pred_low`
- `pred_close`
- `pred_volume`

说明：

- 对于 `Kronos` 官方预测路径，最终会自动把预测列标准化为 `pred_*`
- 如果模型没有输出 `volume`，则图形绘制不受影响，但 CSV 中相关列取决于模型输出

参数搜索模式下（`--tune`）：

- 输出文件改为 `--tune-out` 指定的评分表
- 每行是一组参数组合
- 包含 `mae_close`、`rmse_close`、`mape_close_pct`、`eval_windows`、`status`

### 2. K 线图输出（单张图）

输出文件默认是 `kronos_pred.png`。

图中内容：

- 上半部分：`历史 + 预测` K 线
- 下半部分：`历史 + 真实` K 线（仅在可取到真实未来数据时显示）
- 两个面板都用灰色虚线标识预测起点

无真实未来数据时：

- 只输出上半部分（预测面板），不会额外输出第二张图

## 脚本工作流程

### Kronos 官方模型路径

当 `--model-path` 是一个包含 `config.json + *.safetensors` 的目录时：

1. 从 qlib 读取 OHLCV 历史数据
2. 截取最近 `window` 根 K 线
3. 构造未来 `horizon` 个交易日时间戳
4. 使用 `KronosPredictor.predict(...)` 预测未来走势
5. 使用最后一个可用 `factor` 对预测价格做还原（`pred_price / factor`）
6. 保存 CSV
7. 绘制单张统一 K 线图（上预测下真实）

### 普通 PyTorch 模型路径

当 `--model-path` 是 `pt/pth/ckpt/bin` 文件或包含这些文件的目录时：

1. 从 qlib 读取 OHLCV 历史数据
2. 做 z-score 标准化
3. 构造滑窗样本
4. 批量推理
5. 对结果做反标准化
6. 使用最后一个可用 `factor` 对预测价格做还原（`pred_price / factor`）
7. 保存单张统一 K 线图（上预测下真实）

### 效果
* 预测K线
![预测K线](https://www.fighteryu.xyz/blog/img/6402.png)
* 历史K线对比
![历史对比K线](https://www.fighteryu.xyz/blog/img/6405.png)
#### 欢迎关注
![](https://www.fighteryu.xyz/blog/img/gzh.jpeg)
