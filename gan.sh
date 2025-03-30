# 训练部分
python3 -m minamo.train --epochs 10 --resume true
python3 -m minamo.train --epochs 10 --resume true --train "datasets/minamo-dataset-1.json" --validate "datasets/minamo-eval-1.json"
python3 -m minamo.train --epochs 10 --resume true
python3 -m ginka.train --epochs 30 --resume true
python3 -m ginka.validate
# 训练完毕，处理数据
mv "minamo-dataset.json" "datasets/minamo-dataset-$1.json"
mv "minamo-eval.json" "datasets/minamo-eval-$1.json"
cd data
pnpm minamo "../minamo-dataset.json" "../result/ginka_val.json" "../../Apeiria/project" assigned:100:30
pnpm minamo "../minamo-eval.json" "../result/ginka_val.json" "../../Apeiria-eval/project" assigned:100:10
pnpm merge "../datasets/minamo-dataset-merged.json" "../datasets/minamo-dataset-merged.json" "../datasets/minamo-dataset-$1.json"
pnpm merge "../datasets/minamo-eval-merged.json" "../datasets/minamo-eval-merged.json" "../datasets/minamo-eval-$1.json"
pnpm review "../minamo-dataset.json" "../datasets/minamo-dataset-merged.json"
pnpm review "../minamo-eval.json" "../datasets/minamo-eval-merged.json"
cd ..
