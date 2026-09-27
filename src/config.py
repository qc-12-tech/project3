"""全局配置：路径 / decoder-only 模型超参 / 训练超参 / 生成式 Prompt 模板。

模型是**纯 decoder-only（GPT 式）因果语言模型**，不使用任何 encoder。
情感与评分都以「生成下一个 token」的形式建模：

    <bos> 影评：{评论} \n 情感：{好|差} ␣ 评分：{1..5} <eos>

训练时只在 {好|差} 与 {1..5} 两个位置上计算交叉熵（star==3 的中评屏蔽情感位，
只监督评分位）；推理时用 KV cache 自回归解码两步，取受限词表上的概率分布。
"""
from dataclasses import dataclass, asdict
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    """允许用环境变量重定向产物/数据目录（冒烟测试与全量结果互不干扰）。"""
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


DATA_CSV = _env_path("PROJECT_DATA_CSV", ROOT / "data" / "DMSC.csv")
DATA_DIR = ROOT / "data"
ART_DIR = _env_path("PROJECT_ART_DIR", ROOT / "artifacts")
CACHE_DIR = ART_DIR / "cache"
PRED_DIR = ART_DIR / "preds"

# ---------------------------------------------------------------- 特殊 token
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# ---------------------------------------------------------------- Prompt 模板
PREFIX = "影评："          # 输入前缀
MID = "\n情感："            # 第 1 个生成目标
SEP = " 评分："             # 第 2 个生成目标
POS_CHAR, NEG_CHAR = "好", "差"
RATING_CHARS = ["1", "2", "3", "4", "5"]
TEMPLATE_CHARS = PREFIX + MID + SEP  # 模板中出现的字符，必须进词表

# ---------------------------------------------------------------- 标签口径
STAR_POS_MIN = 4   # 4~5 星 = 好评
STAR_NEG_MAX = 2   # 1~2 星 = 差评
STAR_MID = 3       # 3 星 = 中评（情感位屏蔽，只监督评分位）


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 8
    d_ff: int = 688          # SwiGLU 隐层
    max_len: int = 144       # 覆盖 p99(140)
    dropout: float = 0.1
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 128       # MPS 上 128 比 512 更快（显存压力小）
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 2
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    max_train_samples: int = 150_000
    max_val_samples: int = 15_000
    token_budget: int = 8192      # 批内 token 预算：批大小 = budget / 序列长度（钉住激活内存）
    log_every: int = 25
    seed: int = 42
    device: str = "auto"     # auto -> mps > cuda > cpu

    def to_dict(self):
        return asdict(self)


@dataclass
class AggregateConfig:
    prior_m: int = 100          # 贝叶斯均分的先验条数
    aspect_sample_per_movie: int = 4000   # 每部电影用于优缺点挖掘的抽样条数
    aspect_min_mentions: int = 30         # 方面词进入优缺点的最小提及数
    aspect_pos_threshold: float = 0.60
    aspect_neg_threshold: float = 0.40
    top_aspects: int = 6
    top_keywords: int = 12


def pick_device(prefer: str = "auto") -> str:
    import torch
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
