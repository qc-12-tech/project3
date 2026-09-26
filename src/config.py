"""项目全局配置：路径、标签、模型超参、告警阈值。"""
import os

# ---------------- 路径 ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

RAW_DATA = os.path.join(DATA_DIR, "reviews.jsonl")          # 生成的原始数据
SAMPLE_DATA = os.path.join(DATA_DIR, "sample_1000.jsonl")   # 小样本，快速验证
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.json")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pt")
CACHE_PREFIX = os.path.join(OUTPUT_DIR, "encoded")          # 编码缓存前缀
KEYWORDS_PATH = os.path.join(OUTPUT_DIR, "keywords.json")
DB_PATH = os.path.join(OUTPUT_DIR, "reviews.db")

# ---------------- 标签 ----------------
LABELS = ["好评", "中评", "差评"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

# 打分映射：好评=1.0，中评=0.5，差评=0.0
LABEL_SCORE = [1.0, 0.5, 0.0]

# ---------------- 数据 ----------------
TOTAL_SAMPLES = 1_000_000
# 生成时三类占比：好评 / 中评 / 差评
LABEL_RATIO = [0.45, 0.25, 0.30]
SEED = 42
# 模拟标注噪声：按此比例随机翻转标签，使任务不可能 100% 正确，
# 也让“判断可能错误的概率”具有真实意义
LABEL_NOISE = 0.08
# 边界样本比例：好评/差评中混入轻微反向描述，制造模糊样本
MIX_RATIO = 0.20

# ---------------- 分词/词表 ----------------
MAX_LEN = 64                 # 最大字符长度
MIN_CHAR_FREQ = 2            # 字频低于该值归为 <unk>
PAD_ID = 0
UNK_ID = 1

# ---------------- 模型 ----------------
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 3
D_FF = 256
DROPOUT = 0.15
NUM_CLASSES = len(LABELS)

# ---------------- 训练 ----------------
BATCH_SIZE = 256
EPOCHS = 3
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LABEL_SMOOTHING = 0.05
VAL_RATIO = 0.02
GRAD_CLIP = 1.0

# ---------------- 推理 / 告警阈值 ----------------
MC_SAMPLES = 8               # MC Dropout 前向次数
MC_DROPOUT = 0.15            # 推理时启用的 dropout 比例

# 分数低于该值 -> 严重问题（需立即处理）
SCORE_CRITICAL = 0.30
# 分数低于该值 -> 需要关注
SCORE_WARNING = 0.55
# 判断错误概率高于该值 -> 提醒人工复核
ERROR_ALERT = 0.40

# 关键词提取
KEYWORD_TOP_K = 30
KEYWORD_MIN_COUNT = 5
KEYWORD_MIN_LIFT = 1.3        # 差评占比 / 整体占比 至少多少倍才算“问题词”
KEYWORD_SCAN_LIMIT = 50_000   # 提取关键词时最多扫描多少条差评（-1 表示全部）

# ---------------- 查重 / 刷评论检测 ----------------
DUP_SIM_THRESHOLD = 0.55   # 字符 2-gram Jaccard 相似度阈值，>= 该值视为近似重复
DUP_RATE_HIGH = 0.40       # 重复率 >= 该值 -> 疑似刷评论
DUP_RATE_WARN = 0.20       # 重复率 >= 该值 -> 需关注
