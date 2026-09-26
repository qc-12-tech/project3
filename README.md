# 外卖评价识别系统（Transformer）

基于**从零实现的 Transformer Encoder** 的中文外卖评价分析系统。支持好评/中评/差评三分类、
差评高频问题词挖掘、风险打分与告警、预测错误概率（不确定度）估计、人工复核队列和样本存储。
架构采用 **FastAPI 提供接口 + Streamlit 作为前端**。

## 功能特性

| 需求 | 实现 |
| --- | --- |
| 好评/中评/差评识别 | 字符级 Transformer Encoder 三分类（`src/model.py`） |
| 差评高频词 / 主要问题 | jieba 分词 + 词频 + lift 区分度 + 维度聚合（`src/keywords.py`） |
| 对所有评价打分 | 情感分 `score = Σ pᵢ·sᵢ`（好评1 / 中评0.5 / 差评0），越低越严重 |
| 低于阈值提醒处理 | `score < SCORE_CRITICAL(0.30)` → `need_process=true`，严重程度=严重 |
| 自主输入评价 | `POST /predict`、`POST /predict/batch` |
| 判断可能错误的概率 | MC Dropout 多次前向，`error_prob = 1 - max(prob)`（+ 温度校准） |
| 高于阈值提醒人工判断 | `error_prob >= ERROR_ALERT(0.40)` → `need_review=true`，进入待复核队列 |
| 样本存储供后续判断 | SQLite `outputs/reviews.db`，可回填真实标签用于再训练 |
| 评价查重 / 刷评论检测 | 完全重复 + 字符 n-gram 近似重复识别，判定是否存在刷评（`src/duplicate.py`） |

## 目录结构

```
project3/
├── src/
│   ├── config.py          # 全局配置：路径、标签、超参、告警阈值
│   ├── generate_data.py   # 模拟生成 100 万外卖评价
│   ├── vocab.py           # 字符级词表
│   ├── dataset.py         # Dataset/DataLoader + 磁盘编码缓存(mmap)
│   ├── model.py           # 从零实现 Transformer Encoder 分类器
│   ├── train.py           # 训练脚本
│   ├── retrain.py         # 主动学习增量再训练（消费 true_label）
│   ├── predict.py         # 推理：打分 + MC Dropout 错误概率 + 温度校准
│   ├── duplicate.py       # 评价查重 / 刷评论检测
│   ├── keywords.py        # 差评高频词 / 问题维度分析
│   └── store.py           # SQLite 样本存储与复核队列
├── app/
│   ├── main.py            # FastAPI 接口
│   └── streamlit_app.py   # Streamlit 前端（调用 FastAPI）
├── data/reviews.jsonl     # 生成的数据（text, label）
├── outputs/               # 模型、词表、关键词、数据库、日志
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 生成 100 万条模拟评价（约 7 秒）
python -m src.generate_data --n 1000000 --out data/reviews.jsonl

# 2) 训练（MPS/CPU 自动选择；1M×3 epoch 约 50 分钟，可先用 --max-samples 快速验证）
python -m src.train --data data/reviews.jsonl --epochs 3

# 3) 差评高频词 / 主要问题
python -m src.keywords --data data/reviews.jsonl --scan-limit 50000

# 4) 命令行推理
python -m src.predict --text "等了两个小时，饭都凉了，差评！"

# 5) 温度校准（让“错误概率”更可靠，可选）
python -m src.predict --calibrate --max-samples 20000

# 6) 启动后端接口
uvicorn app.main:app --reload --port 8000
# 打开接口文档 http://127.0.0.1:8000/docs

# 7) 另开一个终端，启动 Streamlit 前端
streamlit run app/streamlit_app.py
# 浏览器会自动打开 http://127.0.0.1:8501
```

快速验证（小数据，几分钟）：

```bash
python -m src.generate_data --n 20000 --out data/sample.jsonl
python -m src.train --data data/sample.jsonl --epochs 2
```

## 判定与告警规则

- **情感分** `score ∈ [0,1]`：`score = 1·P(好评) + 0.5·P(中评) + 0·P(差评)`
- **严重程度**：`score < 0.30` → 严重；`0.30 ≤ score < 0.55` → 警告；否则正常
- **错误概率** `error_prob = 1 - max(P)`（对 MC Dropout 多次结果取平均），
  取值越大说明模型越犹豫、越可能判错
- **需立即处理** `need_process = score < 0.30`
- **需人工复核** `need_review = error_prob ≥ 0.40`
- 阈值集中定义在 `src/config.py`，可自行调整

## 前端界面（Streamlit）

前端通过 HTTP 调用 FastAPI 接口，默认地址 `http://127.0.0.1:8000`（可在左侧栏修改）。包含 6 个页面（侧边栏导航，美团风格）：

- **评价识别**：输入评价 → 展示分类、满意度（1~5 星）、置信度、处理等级与告警，并给出概率柱状图。
- **批量识别**：多行输入 → 汇总各大类数量与告警列表，表格查看全部结果。
- **差评分析**：差评高频问题词柱状图 + 主要问题维度占比 + 各维度代表词。
- **查重检测**：粘贴同一商品的多条评价 → 检测完全重复/近似重复，判定是否疑似刷评论。
- **待核实评价**：待核实队列逐条处理，回填真实标签（写入样本库）或标记已处理。
- **数据统计**：样本总数、待核实/待处理数量、平均满意度与类别分布。

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/predict` | 单条评价预测，返回标签/概率/分数/错误概率/告警 |
| POST | `/predict/batch` | 批量预测，返回汇总与告警列表 |
| GET | `/negative/keywords` | 差评高频词与问题维度报告 |
| POST | `/negative/analyze` | 对传入的差评文本实时分析 |
| GET | `/reviews` | 分页查询样本（支持标签/复核/处理筛选） |
| GET | `/reviews/pending` | 待人工复核队列 |
| POST | `/reviews/{id}/label` | 回填人工真实标签（存入样本库） |
| POST | `/reviews/{id}/process` | 标记问题已处理 |
| POST | `/duplicate/check` | 对多条评价做查重，返回重复率与疑似刷评论判定 |
| GET | `/stats` | 总体统计（各类占比、待处理/待复核数、均分） |

示例：

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text":"味道还行，但是配送有点慢，包装也洒了","save":true}'
```

返回（节选）：

```json
{
  "label": "差评",
  "probs": {"好评": 0.058, "中评": 0.395, "差评": 0.546},
  "score": 0.256,
  "error_prob": 0.454,
  "severity": "严重",
  "need_process": true,
  "need_review": true
}
```

`error_prob=0.454 ≥ 0.40`，说明该条情绪矛盾、模型不确定，自动进入人工复核队列。

## 模型结构

字符 Embedding → 正弦位置编码 → 3 × Transformer Encoder（Post-LN，4 头注意力）
→ Masked Mean Pooling → LayerNorm → Dropout → 线性分类头。约 0.43M 参数，纯 PyTorch，
无需预训练模型。细节见 `src/model.py`，实现风格与 `testcode/transformer.py` 保持一致。

## 查重 / 刷评论检测

判断一个商品是否存在刷评，核心看两类信号：

1. **完全重复**：一字不差的评价出现多次（复制粘贴刷评）；
2. **近似重复**：换几个字 / 加标点的模板化评价（用字符 n-gram 的 Jaccard 相似度做贪心聚类识别）。

重复率 `= 1 - 聚类后唯一评价数 / 总评价数`，阈值在 `src/config.py` 的 `DUP_RATE_HIGH(0.40)` / `DUP_RATE_WARN(0.20)`：

```bash
# 命令行试一下
python -m src.duplicate
```

前端「查重检测」页调用 `POST /duplicate/check`，返回重复率、判定结论，以及完全重复/近似重复的分组明细。

## 主动学习闭环（增量再训练）

系统会把 `error_prob` 高（模型不确定）的样本送入人工复核队列，人工回填真实标签后
写入样本库的 `true_label` 字段。`src/retrain.py` 读取这些人工标注的“困难样本”，
合入训练集做增量微调，形成「低置信 → 人工标注 → 再训练 → 更准」的闭环：

```bash
# 前提：已在前端“人工复核”标签页给若干样本回填了真实标签
python -m src.retrain                          # 默认 1 epoch，lr=3e-5，20 万原始样本
python -m src.retrain --epochs 2 --human-ratio 0.5
python -m src.retrain --max-orig-samples 0     # 使用全部原始样本（更慢但更稳）
```

- 困难样本默认在每个 batch 中占 **30%**（`--human-ratio`），用加权采样上采样，避免被海量原始样本淹没。
- 复用原始语料的磁盘编码缓存（mmap），不会重新编码 1M 语料。
- 微调前后在「困难样本验证集」+「整体回归集」上评估，报告写入 `outputs/retrain_report.json`。
- 写回 `outputs/model.pt` 前会备份旧模型到 `outputs/model.pt.before_retrain`；重启后端后生效。

## 换成真实数据

数据格式为每行一个 JSON：`{"text": "评价内容", "label": 0}`，其中 `label` 取 `0=好评, 1=中评, 2=差评`。
替换 `data/reviews.jsonl` 后重新运行 `src.train` 与 `src.keywords` 即可（首次会重建编码缓存）。
删除 `outputs/encoded_*.npy` 与 `outputs/*_meta.json` 可强制重新编码。

## 说明与可扩展方向

- 模拟数据由短语模板组合生成，并注入约 **8% 标注噪声**与 **20% 边界样本**（好评/差评中混入轻微反向描述），
  因此验证集准确率约 85% 左右属正常，且能让“判断可能错误的概率”真正有意义；真实场景请替换为线上数据。
  可用 `--label-noise`、`--mix-ratio`（见 `src/config.py`）调节任务难度。
- 可扩展：接入预训练中文 BERT 提升精度、把 `error_prob` 高的样本优先排进人工复核队列
  （主动学习采样策略）；增量再训练已由 `src/retrain.py` 实现。
