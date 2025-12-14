# 从头训练
python3 -u -m ginka.train_wgan --epochs 20 --curr_epoch 1 --checkpoint 1 >> output.log
# 接续训练
python3 -u -m ginka.train_wgan --resume true --epochs 300 --state_ginka "result/wgan/ginka-100.pth" --state_minamo "result/wgan/minamo-100.pth" >> output.log

# rnn
python3 -u -m ginka.train_rnn --epochs 150 --checkpoint 10 >> output_rnn.log
python3 -u -m ginka.train_rnn --resume true --epochs 150 --checkpoint 10 --state_ginka "result/rnn/ginka-100.pth" >> output_rnn.log
