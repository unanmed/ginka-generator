# 从头训练
python3 -u -m ginka.train_wgan --epochs 300 >> output.log
# 接续训练
python3 -u -m ginka.train_wgan --resume true --epochs 300 --state_ginka "result/wgan/ginka-100.pth" --state_minamo "result/wgan/minamo-100.pth" >> output.log
