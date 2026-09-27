#!/usr/bin/env bash
# 一键跑通全流程：数据准备 -> 训练 decoder-only 模型 -> 全量推理 -> 优缺点挖掘 -> 刷评论检测 -> 聚合 -> 自检
#
#   bash scripts/run_all.sh              # 全量（数据准备 ~18min + 训练 ~40min + 推理 ~66min + 其余 ~10min）
#   bash scripts/run_all.sh --smoke      # 冒烟：每阶段跑小样本，产物写到 artifacts_smoke/，约 10 分钟
#
# 说明：训练与推理默认用 MPS；本机内存只有 8.6GB，脚本已通过 src/__init__.py 限制显存水位。
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=0
if [ "${1:-}" = "--smoke" ]; then SMOKE=1; shift; fi

if [ "$SMOKE" = "1" ]; then
  export PROJECT_ART_DIR="$PWD/artifacts_smoke"
  export PROJECT_SMOKE=1
  LIMIT=60000
  rm -rf "$PROJECT_ART_DIR"
  echo "==> 冒烟模式：产物目录 ${PROJECT_ART_DIR}，每阶段样本上限 $LIMIT"
  echo "==> 1/7 数据准备"
  python3 -m src.data --limit "$LIMIT"
  echo "==> 2/7 训练（小样本）"
  python3 -m src.train --smoke
  echo "==> 3/7 推理"
  python3 -m src.infer --limit "$LIMIT"
  echo "==> 4/7 优缺点挖掘"
  python3 -m src.aspects --per-movie 300 --limit "$LIMIT"
  echo "==> 5/7 刷评论检测"
  python3 -m src.spam --limit "$LIMIT"
  echo "==> 6/7 聚合"
  python3 -m src.aggregate
  echo "==> 7/7 自检 + 测试"
  python3 scripts/verify.py
  python3 -m pytest -q
  echo "冒烟完成！产物在 artifacts_smoke/（不影响 artifacts/ 的全量结果）"
  exit 0
fi

echo "==> 1/7 数据准备（清洗/去重/编码/缓存）"
python3 -m src.data "$@"

echo "==> 2/7 训练 decoder-only 因果语言模型"
python3 -m src.train

echo "==> 3/7 全量推理（分段续跑 + 增量落盘）"
TOTAL=$(python3 -c "
from src.config import CACHE_DIR
import numpy as np
print(len(np.load(CACHE_DIR / 'predict.off.npy', mmap_mode='r')) - 1)")
STEP=509050
s=0
while [ "$s" -lt "$TOTAL" ]; do
  e=$((s + STEP)); [ "$e" -gt "$TOTAL" ] && e=$TOTAL
  python3 -m src.infer --start "$s" --end "$e" --flush-every 250000
  s=$e
done

echo "==> 4/7 优缺点挖掘（方面词 + log-odds 关键词）"
python3 -m src.aspects

echo "==> 5/7 刷评论（水军）检测"
python3 -m src.spam

echo "==> 6/7 聚合打分与报告"
python3 -m src.aggregate

echo "==> 7/7 自检 + 测试"
python3 scripts/verify.py
python3 -m pytest -q

echo "完成！结果见 artifacts/movie_summary.csv 与 artifacts/分析报告.md"
