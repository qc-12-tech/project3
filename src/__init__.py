"""电影评价分析项目：decoder-only（GPT 式）因果语言模型 + 统计/打分/优缺点挖掘。

在导入 torch 之前先限制 MPS 显存水位：本机统一内存只有 8.6GB，而 PyTorch MPS 默认
允许占用 ~9GB（等于整机内存），会把系统推入 swap 抖动，训练步耗时从 0.7s 飙到 7s。
把水位压到 40%（约 3.4GB）后步耗时稳定。
"""
import os
import sys

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")   # 0 = 不设下界，避免 low>high 报错
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "6")

__all__ = ["config", "vocab", "data", "model", "train", "infer", "aspects", "aggregate", "lexicon"]
