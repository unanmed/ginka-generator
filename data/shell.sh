# 自动处理塔信息
pnpm auto "result.json" "F:/mota-ai/total data/towerinfo.json" "F:\mota-ai\total data/games"
# 将数据按比例区分为训练集和数据集
pnpm eval "ginka-dataset.json" "ginka-eval.json" "result.json" 0.02
